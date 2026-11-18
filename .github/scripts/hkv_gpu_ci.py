# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trusted GitHub control plane for the isolated, pinned HKV GPU pipeline."""

import json
import os
from pathlib import Path
import re
import signal
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


_REPO = "NVIDIA-Merlin/HierarchicalKV"
_REF = "hrong/hkv-ci"
_DEPLOYMENT = "f00e3879cacd5adcc2830f0c669a437670cea872"
_TRUSTED = {"emmaqiaoch", "rhdong", "ranjeet-nvidia", "jiashuy"}
_ACTIVE = {"created", "waiting_for_resource", "preparing", "pending", "running"}
_CONTEXT = "hkv/H100-NVL"
_POLL_SECONDS = 30
_TIMEOUT_SECONDS = 205 * 60
_PIPELINE_PAGES = 50


class _Error(Exception):
    """Only fixed, public-safe messages may cross the control boundary."""


class _Stale(_Error):
    pass


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise _Error("API redirect refused")


class _API:
    def __init__(self, origin, headers):
        address = urlsplit(origin)
        if (
            address.scheme != "https"
            or not address.hostname
            or address.username
            or address.password
            or address.path not in ("", "/")
            or address.query
            or address.fragment
        ):
            raise _Error("Invalid API origin")
        self.origin = origin.rstrip("/")
        self.headers = headers

    def request(self, method, path, payload=None):
        if not path.startswith("/") or path.startswith("//"):
            raise _Error("Invalid API path")
        body = None if payload is None else json.dumps(payload).encode()
        request = Request(
            self.origin + path,
            data=body,
            method=method,
            headers=self.headers | {"Content-Type": "application/json"},
        )
        try:
            with build_opener(_NoRedirect()).open(request, timeout=30) as response:
                data = response.read((1 << 20) + 1)
            if len(data) > 1 << 20:
                raise _Error("API response exceeded limit")
            return json.loads(data)
        except (HTTPError, URLError, OSError, ValueError):
            raise _Error("API request failed") from None


def _id(value):
    if not re.fullmatch(r"[1-9][0-9]{0,19}", str(value)):
        raise _Error("Invalid numeric identity")
    return int(value)


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", value):
        raise _Error("Complete commit identity required")
    return value.lower()


def _gh(path):
    return "/repos/" + _REPO + path


def _trusted(gh, login):
    if not isinstance(login, str) or login.lower() not in _TRUSTED:
        return False
    permission = gh.request(
        "GET", _gh("/collaborators/" + quote(login, safe="") + "/permission")
    )
    return permission.get("permission") in {"write", "maintain", "admin"}


def _pr(gh, number):
    pr = gh.request("GET", _gh("/pulls/" + str(_id(number))))
    if pr.get("base", {}).get("repo", {}).get("full_name") != _REPO:
        raise _Error("Unexpected pull request repository")
    _id(pr.get("number"))
    _sha(pr.get("head", {}).get("sha"))
    return pr


def resolve(gh, event_name, event):
    if event_name == "push":
        if event.get("deleted") or not event.get("ref", "").startswith("refs/heads/"):
            return None
        head = _sha(event.get("after"))
        branch = event["ref"][len("refs/heads/") :]
        query = urlencode(
            {
                "state": "open",
                "head": _REPO.split("/")[0] + ":" + branch,
                "per_page": 100,
            }
        )
        prs = gh.request("GET", _gh("/pulls?") + query)
        candidates = [
            pr
            for pr in prs
            if pr.get("head", {}).get("sha") == head
            and pr.get("head", {}).get("repo", {}).get("full_name") == _REPO
        ]
        if not candidates:
            return None
        pr = _pr(gh, min(_id(pr["number"]) for pr in candidates))
        return pr if _sha(pr["head"]["sha"]) == head else None
    if event_name == "pull_request_target":
        if event.get("action") not in {
            "opened",
            "reopened",
            "ready_for_review",
            "synchronize",
        }:
            return None
        pr = _pr(gh, event["number"])
        return (
            pr
            if _sha(pr["head"]["sha"]) == _sha(event["pull_request"]["head"]["sha"])
            else None
        )
    if event_name == "issue_comment":
        if (
            event.get("action") != "created"
            or not event.get("issue", {}).get("pull_request")
            or not re.fullmatch(
                r"/hkv-gpu-ci [0-9a-fA-F]{40}",
                event.get("comment", {}).get("body", "").strip(),
            )
        ):
            return None
        return _pr(gh, event["issue"]["number"])
    if event_name == "workflow_dispatch":
        return _pr(gh, event.get("inputs", {}).get("pull_request"))
    raise _Error("Unsupported control event")


def _approval(gh, comment, head):
    return comment.get(
        "body", ""
    ).strip().lower() == "/hkv-gpu-ci " + head and _trusted(
        gh, comment.get("user", {}).get("login")
    )


def _admitted(gh, pr, event_name, event):
    actor = os.environ["GITHUB_ACTOR"]
    triggering = os.environ.get("GITHUB_TRIGGERING_ACTOR", actor)
    if not _trusted(gh, actor) or not _trusted(gh, triggering):
        return False
    head = _sha(pr["head"]["sha"])
    if event_name == "issue_comment":
        if event["comment"].get("body", "").strip().lower() != "/hkv-gpu-ci " + head:
            return False
        comment = gh.request(
            "GET", _gh("/issues/comments/" + str(_id(event["comment"]["id"])))
        )
        return comment.get("id") == _id(event["comment"]["id"]) and _approval(
            gh, comment, head
        )
    if _trusted(gh, pr.get("user", {}).get("login")):
        return True
    if event_name == "workflow_dispatch":
        page = 1
        while True:
            comments = gh.request(
                "GET",
                _gh(f"/issues/{_id(pr['number'])}/comments?per_page=100&page={page}"),
            )
            if any(_approval(gh, comment, head) for comment in comments):
                return True
            if len(comments) < 100:
                break
            page += 1
            if page > 10:
                raise _Error("Approval lookup limit exceeded")
    return False


def _status(gh, head, state, description):
    run_id = _id(os.environ["GITHUB_RUN_ID"])
    gh.request(
        "POST",
        _gh("/statuses/" + head),
        {
            "state": state,
            "context": _CONTEXT,
            "description": description,
            "target_url": f"https://github.com/{_REPO}/actions/runs/{run_id}",
        },
    )


def _has_status(gh, head):
    for page in range(1, 11):
        statuses = gh.request(
            "GET", _gh(f"/commits/{head}/statuses?per_page=100&page={page}")
        )
        if any(status.get("context") == _CONTEXT for status in statuses):
            return True
        if len(statuses) < 100:
            return False
    raise _Error("Status lookup limit exceeded")


def _pipeline(gl, prefix, pipeline_id):
    pipeline = gl.request("GET", prefix + "/pipelines/" + str(_id(pipeline_id)))
    if (
        pipeline.get("sha") != _DEPLOYMENT
        or pipeline.get("ref") != _REF
        or pipeline.get("source") != "api"
        or _id(pipeline.get("id")) != pipeline_id
    ):
        raise _Error("Unexpected host pipeline identity")
    return pipeline


def _variables(gl, prefix, pipeline_id, head):
    variables = gl.request("GET", prefix + f"/pipelines/{pipeline_id}/variables")
    return (
        len(variables) == 1
        and variables[0].get("key") == "HKV_COMMIT_SHA"
        and variables[0].get("value") == head
        and variables[0].get("variable_type", "env_var") == "env_var"
    )


def _evidence(gl, prefix, pipeline, head):
    pipeline_id = _id(pipeline["id"])
    if pipeline.get("status") != "success" or not _variables(
        gl, prefix, pipeline_id, head
    ):
        raise _Error("Host pipeline did not prove success")
    bridges = gl.request(
        "GET", prefix + f"/pipelines/{pipeline_id}/bridges?per_page=100"
    )
    jobs = gl.request(
        "GET",
        prefix + f"/pipelines/{pipeline_id}/jobs?include_retried=true&per_page=100",
    )
    if bridges or len(jobs) != 1:
        raise _Error("Unexpected host job topology")
    job = jobs[0]
    job_id = _id(job.get("id"))
    if (
        job.get("name") != "hkv:gpu"
        or job.get("status") != "success"
        or job.get("allow_failure") is not False
        or job.get("commit", {}).get("id") != _DEPLOYMENT
        or job.get("ref") != _REF
        or job.get("pipeline", {}).get("id") != pipeline_id
    ):
        raise _Error("GPU job did not prove success")
    metadata = gl.request(
        "GET", prefix + f"/jobs/{job_id}/artifacts/hkv-results/{job_id}/source.json"
    )
    if (
        metadata.get("repository") != "https://github.com/" + _REPO + ".git"
        or metadata.get("approved_sha") != head
        or metadata.get("job_id") != str(job_id)
        or metadata.get("source_verified") is not True
        or metadata.get("result") != "passed"
        or metadata.get("googletest_repository")
        != "https://github.com/google/googletest.git"
    ):
        raise _Error("GPU source evidence mismatch")
    _sha(metadata.get("googletest_sha"))


def _fresh(gh, number, head):
    current = _pr(gh, number)
    if current.get("state") != "open" or _sha(current["head"]["sha"]) != head:
        raise _Stale("Pull request head changed or closed")


def _interrupt(signum, frame):
    raise _Error("Control interrupted")


def check(gh, pr, event_name, event):
    head, number = _sha(pr["head"]["sha"]), _id(pr["number"])
    if pr.get("state") != "open":
        return
    try:
        admitted = _admitted(gh, pr, event_name, event)
    except Exception:
        if not _has_status(gh, head):
            _status(gh, head, "error", "Trusted admission could not be verified")
        raise _Error("Trusted admission could not be verified") from None
    if not admitted:
        if event_name != "issue_comment" and not _has_status(gh, head):
            _status(
                gh,
                head,
                "pending",
                "Waiting for exact-head trusted maintainer approval",
            )
        return
    _status(gh, head, "pending", "H100 NVL full suite pending")
    gl = None
    prefix = ""
    owned = None
    finished = False
    try:
        deadline = time.monotonic() + _TIMEOUT_SECONDS
        if _sha(os.environ["HKV_GITLAB_DEPLOYMENT_SHA"]) != _DEPLOYMENT:
            raise _Error("Unapproved deployment configuration")
        token = os.environ["HKV_GITLAB_TOKEN"]
        if not token or "\n" in token or "\r" in token:
            raise _Error("Host credential unavailable")
        gl = _API(os.environ["HKV_GITLAB_URL"], {"PRIVATE-TOKEN": token})
        project = _id(os.environ["HKV_GITLAB_PROJECT_ID"])
        prefix = f"/api/v4/projects/{project}"
        branch = gl.request(
            "GET", prefix + "/repository/branches/" + quote(_REF, safe="")
        )
        if branch.get("commit", {}).get("id") != _DEPLOYMENT:
            raise _Error("Host deployment ref moved")
        _fresh(gh, number, head)
        pipeline = None
        retry = (
            event_name in {"issue_comment", "workflow_dispatch"}
            or _id(os.environ.get("GITHUB_RUN_ATTEMPT", "1")) > 1
        )
        reusable = (
            _ACTIVE if retry else _ACTIVE | {"success", "failed", "canceled", "skipped"}
        )
        for page in range(1, _PIPELINE_PAGES + 1):
            if time.monotonic() >= deadline:
                raise _Error("Host lookup exceeded bounded deadline")
            candidates = gl.request(
                "GET",
                prefix
                + "/pipelines?"
                + urlencode(
                    {
                        "ref": _REF,
                        "sha": _DEPLOYMENT,
                        "source": "api",
                        "per_page": 20,
                        "page": page,
                    }
                ),
            )
            for candidate in candidates:
                if time.monotonic() >= deadline:
                    raise _Error("Host lookup exceeded bounded deadline")
                identity = _id(candidate.get("id"))
                if candidate.get("status") in reusable and _variables(
                    gl, prefix, identity, head
                ):
                    pipeline = _pipeline(gl, prefix, identity)
                    break
            if pipeline is not None or len(candidates) < 20:
                break
        else:
            raise _Error("Host history lookup limit exceeded")
        if pipeline is None:
            if time.monotonic() >= deadline:
                raise _Error("Host lookup exceeded bounded deadline")
            created = gl.request(
                "POST",
                prefix + "/pipeline",
                {
                    "ref": _REF,
                    "variables": [
                        {
                            "key": "HKV_COMMIT_SHA",
                            "value": head,
                            "variable_type": "env_var",
                        }
                    ],
                },
            )
            owned = _id(created.get("id"))
            pipeline = _pipeline(gl, prefix, owned)
        while True:
            _fresh(gh, number, head)
            pipeline = _pipeline(gl, prefix, _id(pipeline["id"]))
            state = pipeline.get("status")
            if state == "success":
                _evidence(gl, prefix, pipeline, head)
                _fresh(gh, number, head)
                finished = True
                _status(
                    gh,
                    head,
                    "success",
                    "H100 NVL full suite passed; exact source verified",
                )
                return True
            if state not in _ACTIVE:
                finished = True
                _status(gh, head, "failure", "H100 NVL full suite did not pass")
                return False
            if time.monotonic() >= deadline:
                raise _Error("Host wait exceeded bounded deadline")
            time.sleep(_POLL_SECONDS)
    except _Stale:
        _status(
            gh, head, "error", "Superseded pull request head; GPU result not accepted"
        )
        return False
    except Exception:
        _status(gh, head, "error", "GPU control or evidence error; see Actions run")
        raise _Error("GPU control or evidence error") from None
    finally:
        if gl is not None and owned is not None and not finished:
            try:
                gl.request("POST", prefix + f"/pipelines/{owned}/cancel", {})
            except Exception:
                print(
                    "Owned pipeline cancellation failed; host timeout remains enforced",
                    file=sys.stderr,
                )


def main():
    try:
        if os.environ["GITHUB_REPOSITORY"] != _REPO:
            raise _Error("Unexpected control repository")
        gh = _API(
            "https://api.github.com",
            {
                "Authorization": "Bearer " + os.environ["GITHUB_TOKEN"],
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        raw = Path(os.environ["GITHUB_EVENT_PATH"]).read_bytes()
        if len(raw) > 1 << 20:
            raise _Error("Event exceeded limit")
        event = json.loads(raw)
        event_name = os.environ["GITHUB_EVENT_NAME"]
        pr = resolve(gh, event_name, event)
        if pr is None:
            print("No applicable open pull request")
            return 0
        if sys.argv[1:] == ["resolve"]:
            if pr.get("state") == "open":
                with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
                    output.write(f"pull_request={_id(pr['number'])}\n")
            return 0
        if sys.argv[1:] != ["check"]:
            raise _Error("Expected resolve or check")
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, _interrupt)
        return 1 if check(gh, pr, event_name, event) is False else 0
    except Exception:
        print("HKV GPU control failed; no passing result established", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

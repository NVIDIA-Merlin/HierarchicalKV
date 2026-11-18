# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the actual HTTP and event boundaries without credentials or networking."""

import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "hkv_gpu_ci", ROOT / ".github/scripts/hkv_gpu_ci.py"
)
ci = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ci)
HEAD = "a" * 40
OTHER = "b" * 40
DEPLOYMENT = ci._DEPLOYMENT


class HTTPFixture:
    def __init__(self):
        self.pr = {
            "number": 262,
            "state": "open",
            "user": {"login": "rhdong"},
            "base": {"repo": {"full_name": ci._REPO}},
            "head": {"sha": HEAD, "ref": "feature", "repo": {"full_name": ci._REPO}},
        }
        self.permission = "write"
        self.branch = DEPLOYMENT
        self.pipeline = {
            "id": 11,
            "sha": DEPLOYMENT,
            "ref": ci._REF,
            "source": "api",
            "status": "success",
        }
        self.variables = [
            {"key": "HKV_COMMIT_SHA", "value": HEAD, "variable_type": "env_var"}
        ]
        self.jobs = [
            {
                "id": 22,
                "name": "hkv:gpu",
                "status": "success",
                "allow_failure": False,
                "commit": {"id": DEPLOYMENT},
                "ref": ci._REF,
                "pipeline": {"id": 11},
            }
        ]
        self.bridges = []
        self.metadata = {
            "repository": "https://github.com/" + ci._REPO + ".git",
            "approved_sha": HEAD,
            "job_id": "22",
            "source_verified": True,
            "result": "passed",
            "googletest_repository": "https://github.com/google/googletest.git",
            "googletest_sha": "c" * 40,
        }
        self.comment = {
            "id": 44,
            "body": "/hkv-gpu-ci " + HEAD,
            "user": {"login": "jiashuy"},
        }
        self.existing = []
        self.pipeline_pages = {}
        self.variables_by_pipeline = {}
        self.statuses = []
        self.statuses_by_head = {}
        self.created = []
        self.canceled = []
        self.requests = []
        self.fault = None

    def open(self, request, timeout):
        self.requests.append(request)
        if self.fault is not None:
            self.fault(request)
        path = urlsplit(request.full_url).path
        method = request.get_method()
        payload = json.loads(request.data) if request.data is not None else None
        if request.full_url.startswith("https://api.github.com/"):
            self.assert_token(request, "Authorization", "Bearer gh-fixture")
            if path.endswith("/permission"):
                result = {"permission": self.permission}
            elif "/statuses/" in path:
                self.statuses.append(payload)
                self.statuses_by_head.setdefault(path.rsplit("/", 1)[-1], []).insert(
                    0, payload
                )
                result = payload
            elif path.endswith("/statuses"):
                page = int(parse_qs(urlsplit(request.full_url).query)["page"][0])
                statuses = self.statuses_by_head.get(path.split("/")[-2], [])
                result = statuses[(page - 1) * 100 : page * 100]
            elif path.endswith("/issues/comments/44"):
                result = self.comment
            elif path.endswith("/issues/262/comments"):
                result = [self.comment]
            elif path.endswith("/pulls/262"):
                result = self.pr
            elif path.endswith("/pulls"):
                result = [self.pr]
            else:
                raise AssertionError("unexpected GitHub request")
        else:
            self.assert_token(request, "Private-token", "gl-fixture")
            if "/repository/branches/" in path:
                result = {"commit": {"id": self.branch}}
            elif path.endswith("/pipelines"):
                page = int(
                    parse_qs(urlsplit(request.full_url).query).get("page", ["1"])[0]
                )
                result = self.pipeline_pages.get(page, self.existing)
            elif path.endswith("/pipeline") and method == "POST":
                self.created.append(payload)
                result = self.pipeline
            elif path.endswith("/variables"):
                result = self.variables_by_pipeline.get(
                    int(path.split("/")[-2]), self.variables
                )
            elif path.endswith("/bridges"):
                result = self.bridges
            elif path.endswith("/jobs"):
                result = self.jobs
            elif path.endswith("/source.json"):
                result = self.metadata
            elif path.endswith("/cancel"):
                self.canceled.append(int(path.split("/")[-2]))
                result = {}
            elif path.endswith("/pipelines/11"):
                result = self.pipeline
            else:
                raise AssertionError("unexpected host request")
        return io.BytesIO(json.dumps(result).encode())

    @staticmethod
    def assert_token(request, header, value):
        if request.get_header(header) != value:
            raise AssertionError("wrong authentication domain")


class GPUControlTests(unittest.TestCase):
    def setUp(self):
        self.fixture = HTTPFixture()
        self.env = {
            "GITHUB_REPOSITORY": ci._REPO,
            "GITHUB_TOKEN": "gh-fixture",
            "GITHUB_ACTOR": "EmmaQiaoCh",
            "GITHUB_RUN_ID": "1001",
            "GITHUB_EVENT_NAME": "pull_request_target",
            "HKV_GITLAB_TOKEN": "gl-fixture",
            "HKV_GITLAB_URL": "https://private-host.invalid",
            "HKV_GITLAB_PROJECT_ID": "302940",
            "HKV_GITLAB_DEPLOYMENT_SHA": DEPLOYMENT,
        }
        self.environment = patch.dict(os.environ, self.env, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.transport = patch.object(
            ci, "build_opener", return_value=Mock(open=self.fixture.open)
        )
        self.transport.start()
        self.addCleanup(self.transport.stop)
        self.gh = ci._API(
            "https://api.github.com", {"Authorization": "Bearer gh-fixture"}
        )
        self.event = {
            "number": 262,
            "action": "opened",
            "pull_request": {"head": {"sha": HEAD}},
        }

    def check(self, event_name="pull_request_target", event=None):
        return ci.check(
            self.gh, copy.deepcopy(self.fixture.pr), event_name, event or self.event
        )

    def test_success_requires_terminal_gpu_job_and_exact_artifact(self):
        self.assertTrue(self.check())
        self.assertEqual(
            self.fixture.created,
            [
                {
                    "ref": ci._REF,
                    "variables": [
                        {
                            "key": "HKV_COMMIT_SHA",
                            "value": HEAD,
                            "variable_type": "env_var",
                        }
                    ],
                }
            ],
        )
        self.assertEqual(
            [s["state"] for s in self.fixture.statuses], ["pending", "success"]
        )
        self.assertTrue(
            all(s["context"] == "hkv/H100-NVL" for s in self.fixture.statuses)
        )
        self.assertTrue(
            all(
                s["target_url"]
                == "https://github.com/" + ci._REPO + "/actions/runs/1001"
                for s in self.fixture.statuses
            )
        )
        self.assertEqual(self.fixture.canceled, [])

    def test_untrusted_or_revoked_actor_cannot_dispatch(self):
        for login, permission in (("outsider", "admin"), ("EmmaQiaoCh", "read")):
            with (
                self.subTest(login=login),
                patch.dict(os.environ, {"GITHUB_ACTOR": login}),
            ):
                self.fixture.permission = permission
                self.check()
        self.assertEqual(self.fixture.created, [])
        self.assertTrue(all(s["state"] == "pending" for s in self.fixture.statuses))

    def test_untrusted_rerun_actor_cannot_reuse_original_actor_authority(self):
        with patch.dict(os.environ, {"GITHUB_TRIGGERING_ACTOR": "outsider"}):
            self.check()
        self.assertEqual(self.fixture.created, [])

    def test_same_head_denial_preserves_genuine_success(self):
        self.assertTrue(self.check())
        established = copy.deepcopy(self.fixture.statuses)
        for action in ("ready_for_review", "reopened"):
            with (
                self.subTest(action=action),
                patch.dict(os.environ, {"GITHUB_ACTOR": "outsider"}),
            ):
                self.check(event=self.event | {"action": action})
            self.assertEqual(self.fixture.statuses, established)
            self.assertEqual(len(self.fixture.created), 1)

    def test_denial_preserves_existing_states_but_not_another_heads_status(self):
        for state in ("pending", "success", "failure", "error"):
            ci._status(self.gh, HEAD, state, "Existing exact-head result")
            before = copy.deepcopy(self.fixture.statuses)
            with (
                self.subTest(state=state),
                patch.dict(os.environ, {"GITHUB_ACTOR": "outsider"}),
            ):
                self.check()
            self.assertEqual(self.fixture.statuses, before)
        self.fixture.pr["head"]["sha"] = OTHER
        with patch.dict(os.environ, {"GITHUB_ACTOR": "outsider"}):
            self.check()
        self.assertEqual(self.fixture.statuses_by_head[OTHER][0]["state"], "pending")
        self.assertEqual(self.fixture.statuses_by_head[HEAD][0]["state"], "error")
        self.assertEqual(self.fixture.created, [])

    def test_unverifiable_admission_preserves_existing_same_head_verdict(self):
        ci._status(self.gh, HEAD, "success", "Existing exact-head result")
        established = copy.deepcopy(self.fixture.statuses)

        def unavailable(request):
            if request.full_url.endswith("/permission"):
                raise HTTPError(request.full_url, 503, "unavailable", {}, None)

        self.fixture.fault = unavailable
        with self.assertRaises(ci._Error):
            self.check()
        self.assertEqual(self.fixture.statuses, established)
        self.assertEqual(self.fixture.created, [])

    def test_external_author_requires_exact_fresh_trusted_comment(self):
        self.fixture.pr["user"]["login"] = "outsider"
        self.check()
        self.assertEqual(self.fixture.created, [])
        event = {
            "action": "created",
            "issue": {"number": 262, "pull_request": {}},
            "comment": self.fixture.comment,
        }
        self.assertTrue(self.check("issue_comment", event))
        self.assertEqual(len(self.fixture.created), 1)

    def test_stale_or_untrusted_comment_never_dispatches_or_resets_status(self):
        event = {
            "action": "created",
            "comment": {"id": 44, "body": "/hkv-gpu-ci " + HEAD},
        }
        for body, user in (
            ("/hkv-gpu-ci " + OTHER, "jiashuy"),
            ("/hkv-gpu-ci " + HEAD, "outsider"),
        ):
            self.fixture.comment.update(body=body, user={"login": user})
            self.check("issue_comment", event)
        self.assertEqual(self.fixture.created, [])
        self.assertEqual(self.fixture.statuses, [])

    def test_mutated_comment_cannot_transfer_old_event_approval_to_new_head(self):
        event = {
            "action": "created",
            "comment": {"id": 44, "body": "/hkv-gpu-ci " + OTHER},
        }
        self.check("issue_comment", event)
        self.assertEqual(self.fixture.created, [])
        self.assertEqual(self.fixture.statuses, [])

    def test_manual_external_retry_requires_existing_head_bound_approval(self):
        self.fixture.pr["user"]["login"] = "outsider"
        self.fixture.comment["body"] = "/hkv-gpu-ci " + OTHER
        self.check("workflow_dispatch")
        self.assertEqual(self.fixture.created, [])
        self.fixture.comment["body"] = "/hkv-gpu-ci " + HEAD
        self.assertTrue(self.check("workflow_dispatch"))

    def test_push_resolves_open_pr_and_all_synchronize_events_remain_supported(self):
        event = {"ref": "refs/heads/feature", "after": HEAD}
        self.assertEqual(ci.resolve(self.gh, "push", event)["number"], 262)
        query = parse_qs(urlsplit(self.fixture.requests[0].full_url).query)
        self.assertEqual(query["head"], ["NVIDIA-Merlin:feature"])
        for source in (ci._REPO, "outsider/fork"):
            self.fixture.pr["head"]["repo"]["full_name"] = source
            self.assertEqual(
                ci.resolve(
                    self.gh,
                    "pull_request_target",
                    self.event | {"action": "synchronize"},
                )["number"],
                262,
            )
        self.assertIsNone(
            ci.resolve(self.gh, "push", {"ref": "refs/tags/v1", "after": HEAD})
        )

    def test_stale_automatic_event_does_not_authorize_newer_head(self):
        self.fixture.pr["head"]["sha"] = OTHER
        self.assertIsNone(ci.resolve(self.gh, "pull_request_target", self.event))
        self.assertIsNone(
            ci.resolve(self.gh, "push", {"ref": "refs/heads/feature", "after": HEAD})
        )

    def test_duplicate_automatic_event_reuses_success_with_full_evidence(self):
        self.fixture.existing = [self.fixture.pipeline]
        self.assertTrue(self.check())
        self.assertEqual(self.fixture.created, [])
        self.assertTrue(
            any("/source.json" in req.full_url for req in self.fixture.requests)
        )

    def test_automatic_failure_is_reused_but_explicit_retry_dispatches(self):
        self.fixture.pipeline["status"] = "failed"
        self.fixture.existing = [self.fixture.pipeline]
        self.assertFalse(self.check())
        self.assertEqual(self.fixture.created, [])
        self.assertEqual(self.fixture.statuses[-1]["state"], "failure")
        self.check("workflow_dispatch")
        self.assertEqual(len(self.fixture.created), 1)

    def test_trusted_actions_rerun_is_an_explicit_completed_head_retry(self):
        self.fixture.pipeline["status"] = "failed"
        self.fixture.existing = [self.fixture.pipeline]
        self.assertFalse(self.check())
        self.assertEqual(self.fixture.created, [])
        with patch.dict(os.environ, {"GITHUB_RUN_ATTEMPT": "2"}):
            self.assertFalse(self.check())
        self.assertEqual(len(self.fixture.created), 1)
        with patch.dict(
            os.environ,
            {"GITHUB_RUN_ATTEMPT": "3", "GITHUB_TRIGGERING_ACTOR": "outsider"},
        ):
            self.check()
        self.assertEqual(len(self.fixture.created), 1)

    def test_later_history_pages_reuse_active_success_and_failure(self):
        first = [{"id": identity, "status": "success"} for identity in range(101, 121)]
        for item in first:
            self.fixture.variables_by_pipeline[item["id"]] = [
                {"key": "HKV_COMMIT_SHA", "value": OTHER}
            ]
        for state in ("running", "success", "failed"):
            self.fixture.pipeline["status"] = state
            self.fixture.pipeline_pages = {1: first, 2: [self.fixture.pipeline]}
            self.fixture.requests.clear()
            self.fixture.created.clear()
            self.fixture.canceled.clear()
            with self.subTest(state=state):
                if state == "running":
                    with (
                        patch.object(
                            ci.time, "sleep", side_effect=ci._Error("Stop observer")
                        ),
                        self.assertRaises(ci._Error),
                    ):
                        self.check()
                else:
                    self.assertEqual(self.check(), state == "success")
                self.assertEqual(self.fixture.created, [])
                self.assertEqual(self.fixture.canceled, [])
                pages = [
                    parse_qs(urlsplit(req.full_url).query).get("page", ["1"])[0]
                    for req in self.fixture.requests
                    if urlsplit(req.full_url).path.endswith("/pipelines")
                ]
                self.assertEqual(pages, ["1", "2"])

    def test_truncated_host_history_fails_closed_without_creation(self):
        self.fixture.existing = [
            {"id": identity, "status": "success"} for identity in range(101, 121)
        ]
        for item in self.fixture.existing:
            self.fixture.variables_by_pipeline[item["id"]] = [
                {"key": "HKV_COMMIT_SHA", "value": OTHER}
            ]
        with self.assertRaises(ci._Error):
            self.check()
        self.assertEqual(self.fixture.created, [])
        self.assertEqual(self.fixture.statuses[-1]["state"], "error")

    def test_missing_host_credential_cannot_dispatch_or_pass(self):
        with (
            patch.dict(os.environ, {"HKV_GITLAB_TOKEN": ""}),
            self.assertRaises(ci._Error),
        ):
            self.check()
        self.assertEqual(self.fixture.created, [])
        self.assertEqual(
            [status["state"] for status in self.fixture.statuses], ["pending", "error"]
        )

    def test_different_hkv_variable_is_never_reused(self):
        self.fixture.existing = [self.fixture.pipeline]
        calls = 0

        def varied(request):
            nonlocal calls
            if request.full_url.endswith("/variables"):
                calls += 1
                self.fixture.variables[0]["value"] = OTHER if calls == 1 else HEAD

        self.fixture.fault = varied
        self.assertTrue(self.check())
        self.assertEqual(len(self.fixture.created), 1)

    def test_deployment_pin_and_ref_reject_before_creation(self):
        for variable, branch in ((OTHER, DEPLOYMENT), (DEPLOYMENT, OTHER)):
            with (
                self.subTest(variable=variable),
                patch.dict(os.environ, {"HKV_GITLAB_DEPLOYMENT_SHA": variable}),
            ):
                self.fixture.branch = branch
                with self.assertRaises(ci._Error):
                    self.check()
        self.assertEqual(self.fixture.created, [])

    def test_wrong_pipeline_identity_cancels_only_new_owned_pipeline(self):
        for key, value in (("sha", OTHER), ("source", "web"), ("ref", "master")):
            original = self.fixture.pipeline[key]
            self.fixture.pipeline[key] = value
            with self.subTest(key=key), self.assertRaises(ci._Error):
                self.check()
            self.fixture.pipeline[key] = original
        self.assertEqual(self.fixture.canceled, [11, 11, 11])

    def test_job_topology_and_artifact_mismatch_cannot_pass(self):
        for change in (
            "bridge",
            "extra-job",
            "failed-job",
            "wrong-commit",
            "wrong-job-id",
            "wrong-sha",
            "unverified",
            "failed-result",
            "wrong-repository",
            "bad-gtest",
        ):
            saved = copy.deepcopy(
                (self.fixture.bridges, self.fixture.jobs, self.fixture.metadata)
            )
            if change == "bridge":
                self.fixture.bridges = [{"id": 33}]
            elif change == "extra-job":
                self.fixture.jobs *= 2
            elif change == "failed-job":
                self.fixture.jobs[0]["status"] = "failed"
            elif change == "wrong-commit":
                self.fixture.jobs[0]["commit"]["id"] = OTHER
            elif change == "wrong-job-id":
                self.fixture.metadata["job_id"] = "23"
            elif change == "wrong-sha":
                self.fixture.metadata["approved_sha"] = OTHER
            elif change == "unverified":
                self.fixture.metadata["source_verified"] = False
            elif change == "failed-result":
                self.fixture.metadata["result"] = "failed"
            elif change == "wrong-repository":
                self.fixture.metadata["repository"] = "https://example.invalid/repo.git"
            else:
                self.fixture.metadata["googletest_sha"] = "main"
            with self.subTest(change=change), self.assertRaises(ci._Error):
                self.check()
            self.assertEqual(self.fixture.statuses[-1]["state"], "error")
            self.fixture.bridges, self.fixture.jobs, self.fixture.metadata = saved

    def test_missing_artifact_fails_without_leaking_backend_error(self):
        def absent(request):
            if "/artifacts/" in request.full_url:
                raise HTTPError(
                    request.full_url,
                    404,
                    "SECRET-FIXTURE",
                    {},
                    io.BytesIO(b"SECRET-FIXTURE"),
                )

        self.fixture.fault = absent
        with self.assertRaises(ci._Error) as failure:
            self.check()
        self.assertNotIn("SECRET", str(failure.exception))
        self.assertNotIn("private-host", str(failure.exception))
        self.assertEqual(self.fixture.statuses[-1]["state"], "error")

    def test_timeout_and_interruption_cancel_owned_work(self):
        self.fixture.pipeline["status"] = "running"
        with (
            patch.object(
                ci.time, "monotonic", side_effect=[0, 0, 0, ci._TIMEOUT_SECONDS + 1]
            ),
            self.assertRaises(ci._Error),
        ):
            self.check()
        self.assertEqual(self.fixture.canceled, [11])
        with (
            patch.object(
                ci.time, "sleep", side_effect=ci._Error("Control interrupted")
            ),
            self.assertRaises(ci._Error),
        ):
            self.check()
        self.assertEqual(self.fixture.canceled, [11, 11])
        with self.assertRaises(ci._Error):
            ci._interrupt(signal.SIGTERM, None)

    def test_reused_active_pipeline_is_not_canceled_by_observer_timeout(self):
        self.fixture.pipeline["status"] = "running"
        self.fixture.existing = [self.fixture.pipeline]
        with (
            patch.object(
                ci.time, "monotonic", side_effect=[0, 0, 0, ci._TIMEOUT_SECONDS + 1]
            ),
            self.assertRaises(ci._Error),
        ):
            self.check()
        self.assertEqual(self.fixture.created, [])
        self.assertEqual(self.fixture.canceled, [])

    def test_stale_head_during_polling_cancels_owned_pipeline(self):
        def stale(request):
            if request.full_url.endswith("/pipelines/11"):
                self.fixture.pr["head"]["sha"] = OTHER

        self.fixture.fault = stale
        self.assertFalse(self.check())
        self.assertEqual(self.fixture.canceled, [11])
        self.assertEqual(self.fixture.statuses[-1]["state"], "error")

    def test_transport_rejects_redirects_and_unsafe_origins(self):
        with self.assertRaises(ci._Error):
            ci._NoRedirect().redirect_request(
                None, None, 302, "", {}, "https://other.invalid"
            )
        for origin in (
            "http://host.invalid",
            "https://user:SECRET@host.invalid",
            "https://host.invalid/private",
            "https://host.invalid?token=SECRET",
        ):
            with self.subTest(origin=origin), self.assertRaises(ci._Error):
                ci._API(origin, {})

    def test_cli_errors_are_secret_free_and_resolve_output_is_numeric(self):
        with tempfile.TemporaryDirectory() as directory:
            event = Path(directory) / "event.json"
            output = Path(directory) / "output"
            event.write_text(json.dumps(self.event))
            with (
                patch.dict(
                    os.environ,
                    {"GITHUB_EVENT_PATH": str(event), "GITHUB_OUTPUT": str(output)},
                ),
                patch.object(sys, "argv", ["control", "resolve"]),
            ):
                self.assertEqual(ci.main(), 0)
            self.assertEqual(output.read_text(), "pull_request=262\n")

            def secret_error(request):
                raise HTTPError(request.full_url, 403, "SECRET-FIXTURE", {}, None)

            self.fixture.fault = secret_error
            error = io.StringIO()
            with (
                patch.dict(os.environ, {"GITHUB_EVENT_PATH": str(event)}),
                patch.object(sys, "argv", ["control", "check"]),
                patch("sys.stderr", error),
            ):
                self.assertEqual(ci.main(), 1)
            self.assertEqual(
                error.getvalue(),
                "HKV GPU control failed; no passing result established\n",
            )


if __name__ == "__main__":
    unittest.main()

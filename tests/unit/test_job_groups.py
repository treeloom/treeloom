"""Unit tests for the JobGroup domain model and Job.group_id round-trip.

The Postgres stores (JobGroupStore / JobStore) need a live DB and are verified
by an integration run against a throwaway Postgres; these unit tests cover the
pure serialization that the stores rely on.
"""

from treeloom.domain.jobs import Job, JobGroup, JobStatus


class TestJobGroupSerialization:
    def test_to_dict_round_trips(self):
        g = JobGroup(id="grp_1", label="featbit", kind="fleet",
                     created_at=1000.0, created_by="user_1", task_count=25)
        d = g.to_dict()
        assert d == {
            "id": "grp_1", "label": "featbit", "kind": "fleet",
            "created_at": 1000.0, "created_by": "user_1", "task_count": 25,
        }
        assert JobGroup.from_dict(d) == g

    def test_from_dict_defaults(self):
        g = JobGroup.from_dict({"id": "grp_x", "label": "x"})
        assert g.kind == "repo"          # default kind
        assert g.created_by is None
        assert g.task_count == 0
        assert g.created_at > 0          # default_factory time.time()

    def test_kind_falls_back_to_repo_on_empty(self):
        assert JobGroup.from_dict({"id": "g", "label": "l", "kind": ""}).kind == "repo"


class TestJobGroupIdRoundTrip:
    def test_group_id_defaults_none(self):
        assert Job(id="j1").group_id is None
        assert Job(id="j1").to_dict()["group_id"] is None

    def test_group_id_survives_to_dict_from_dict(self):
        j = Job(id="j1", source_id="s1", status=JobStatus.QUEUED, group_id="grp_42")
        d = j.to_dict()
        assert d["group_id"] == "grp_42"
        assert Job.from_dict(d).group_id == "grp_42"

    def test_from_dict_without_group_id_is_none(self):
        # Legacy dicts (earlier) have no group_id key.
        assert Job.from_dict({"id": "j1", "status": "done"}).group_id is None

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.files import router


@pytest.fixture
def sandbox(tmp_path):
    root = tmp_path / "sandbox"
    root.mkdir()
    return root


@pytest.fixture
def client(sandbox):
    app = FastAPI()
    app.include_router(router)
    app.state.sandbox_root = sandbox
    return TestClient(app)


def test_list_root_shows_folders_first_then_files(client, sandbox):
    (sandbox / "b.txt").write_text("hello")
    (sandbox / "a-folder").mkdir()

    response = client.get("/files")

    assert response.status_code == 200
    assert response.json() == [
        {"name": "a-folder", "type": "dir", "size": None},
        {"name": "b.txt", "type": "file", "size": 5},
    ]


def test_list_nested_folder(client, sandbox):
    (sandbox / "reports").mkdir()
    (sandbox / "reports" / "q1.csv").write_text("x")

    response = client.get("/files", params={"path": "reports"})

    assert response.status_code == 200
    assert [e["name"] for e in response.json()] == ["q1.csv"]


def test_list_a_file_path_is_rejected(client, sandbox):
    (sandbox / "notes.txt").write_text("x")

    response = client.get("/files", params={"path": "notes.txt"})

    assert response.status_code == 400


def test_list_missing_folder_is_404(client):
    response = client.get("/files", params={"path": "nope"})

    assert response.status_code == 404


def test_create_folder(client, sandbox):
    response = client.post("/files/dir", json={"path": "reports/2026"})

    assert response.status_code == 201
    assert (sandbox / "reports" / "2026").is_dir()


def test_create_existing_folder_is_409(client, sandbox):
    (sandbox / "reports").mkdir()

    response = client.post("/files/dir", json={"path": "reports"})

    assert response.status_code == 409


def test_upload_files_into_folder(client, sandbox):
    (sandbox / "in").mkdir()

    response = client.post(
        "/files/upload",
        params={"path": "in"},
        files=[("files", ("a.txt", b"aaa")), ("files", ("b.txt", b"bb"))],
    )

    assert response.status_code == 201
    assert (sandbox / "in" / "a.txt").read_bytes() == b"aaa"
    assert (sandbox / "in" / "b.txt").read_bytes() == b"bb"


def test_upload_strips_directory_parts_from_filename(client, sandbox):
    response = client.post(
        "/files/upload",
        files=[("files", ("../../evil.txt", b"x"))],
    )

    assert response.status_code == 201
    assert (sandbox / "evil.txt").read_bytes() == b"x"
    assert not (sandbox.parent / "evil.txt").exists()


def test_upload_refuses_to_overwrite_existing_file(client, sandbox):
    (sandbox / "a.txt").write_text("original")

    response = client.post("/files/upload", files=[("files", ("a.txt", b"new"))])

    assert response.status_code == 409
    assert (sandbox / "a.txt").read_text() == "original"


def test_download_returns_file_as_attachment(client, sandbox):
    (sandbox / "notes.txt").write_text("hello")

    response = client.get("/files/download", params={"path": "notes.txt"})

    assert response.status_code == 200
    assert response.content == b"hello"
    assert "attachment" in response.headers["content-disposition"]


def test_download_missing_file_is_404(client):
    response = client.get("/files/download", params={"path": "nope.txt"})

    assert response.status_code == 404


@pytest.mark.parametrize("method,url,kwargs", [
    ("get", "/files", {"params": {"path": "../"}}),
    ("get", "/files/download", {"params": {"path": "../secret.txt"}}),
    ("post", "/files/dir", {"json": {"path": "../outside"}}),
    ("post", "/files/upload", {"params": {"path": "../"}, "files": [("files", ("a.txt", b"x"))]}),
])
def test_paths_outside_sandbox_are_rejected(client, sandbox, method, url, kwargs):
    (sandbox.parent / "secret.txt").write_text("secret")

    response = getattr(client, method)(url, **kwargs)

    assert response.status_code == 400
    assert not (sandbox.parent / "outside").exists()
    assert not (sandbox.parent / "a.txt").exists()

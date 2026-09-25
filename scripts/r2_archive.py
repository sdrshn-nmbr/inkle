# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3>=1.40"]
# ///
"""Archive a local artifact directory to Cloudflare R2 as verified tar.zst parts.

upload   pack the directory into ~2 GiB tar.zst parts, upload, verify, and write the reference JSON
verify   re-check every archived object in R2 against the reference
restore  list or restore selected paths from R2 into a new directory
cleanup  re-verify, confirm the local inventory is unchanged, delete the archived local files,
         and leave a pointer README in their place
"""

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tarfile
import tempfile
import threading
import time
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

PART = 64 * 1024 * 1024
SHARD = 2 * 1024**3
CREDENTIALS = Path.home() / ".config/axport/r2.env"
FENCE = chr(96) * 3
TICK = chr(96)
LOCK = threading.Lock()
START = time.time()


def log(event, **fields):
    with LOCK:
        print(json.dumps({"event": event, "elapsed_seconds": round(time.time() - START), **fields}), flush=True)


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def client_for(credentials_path):
    creds = {
        key.strip(): value.strip().strip("\"'")
        for key, value in (
            line.split("=", 1)
            for line in credentials_path.read_text().splitlines()
            if "=" in line and not line.lstrip().startswith("#")
        )
    }
    endpoint = f"https://{creds['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name="auto",
        aws_access_key_id=creds["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=creds["R2_SECRET_ACCESS_KEY"],
        config=Config(
            retries={"max_attempts": 8, "mode": "standard"},
            max_pool_connections=16,
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )
    return client, endpoint


def inventory(root):
    entries = []
    for directory, dirs, files in os.walk(root):
        for name in sorted(dirs + files):
            path = Path(directory) / name
            s = path.lstat()
            if not (stat.S_ISREG(s.st_mode) or stat.S_ISDIR(s.st_mode) or stat.S_ISLNK(s.st_mode)):
                raise RuntimeError(f"UNSUPPORTED_FILE_TYPE: {path}")
            entry = {"path": path.relative_to(root).as_posix(), "size": s.st_size, "mtime_ns": s.st_mtime_ns, "mode": s.st_mode}
            if path.is_symlink():
                entry["target"] = os.readlink(path)
            entries.append(entry)
    return sorted(entries, key=lambda e: e["path"])


def signatures(path):
    sha = hashlib.sha256()
    pieces = []
    with path.open("rb") as handle:
        while chunk := handle.read(PART):
            sha.update(chunk)
            pieces.append(hashlib.md5(chunk).digest())
    if path.stat().st_size >= PART:
        etag = hashlib.md5(b"".join(pieces)).hexdigest() + "-" + str(len(pieces))
    else:
        etag = pieces[0].hex() if pieces else hashlib.md5(b"").hexdigest()
    return sha.hexdigest(), etag


def remote_matches(client, bucket, descriptor):
    head = client.head_object(Bucket=bucket, Key=descriptor["key"])
    return (head["ContentLength"], head["ETag"].strip('"'), head["Metadata"].get("sha256")) == (
        descriptor["bytes"],
        descriptor["etag"],
        descriptor["sha256"],
    )


def upload_verified(client, bucket, path, key):
    sha, etag = signatures(path)
    size = path.stat().st_size
    transfer = TransferConfig(multipart_threshold=PART, multipart_chunksize=PART, max_concurrency=4)
    client.upload_file(str(path), bucket, key, ExtraArgs={"Metadata": {"sha256": sha}}, Config=transfer)
    descriptor = {"key": key, "bytes": size, "sha256": sha, "etag": etag}
    if not remote_matches(client, bucket, descriptor):
        raise RuntimeError(f"REMOTE_VERIFICATION_FAILED: {key}")
    return descriptor


def batches_of(entries):
    batches, current, size = [], [], 0
    for entry in entries:
        entry_size = entry["size"] if stat.S_ISREG(entry["mode"]) else 0
        if current and size + entry_size > SHARD:
            batches.append(current)
            current, size = [], 0
        current.append(entry)
        size += entry_size
    if current:
        batches.append(current)
    return batches


def pack_upload(client, bucket, prefix, root, work, number, entries):
    name = f"part-{number:04d}.tar.zst"
    receipt_path = work / f"{name}.receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if not remote_matches(client, bucket, receipt):
            raise RuntimeError(f"RESUME_REMOTE_MISMATCH: {name}")
        log("resumed", part=name)
        return receipt
    names = work / f"{name}.names"
    names.write_bytes(b"".join(os.fsencode(e["path"]) + b"\0" for e in entries))
    path = work / name
    errors = work / f"{name}.stderr"
    with errors.open("wb") as err, path.open("wb") as out:
        env = {**os.environ, "COPYFILE_DISABLE": "1"}
        tar = subprocess.Popen(
            ["tar", "--no-mac-metadata", "--no-xattrs", "--no-acls", "--no-fflags", "-c", "-f", "-", "--no-recursion", "--null", "-T", str(names)],
            cwd=root, stdout=subprocess.PIPE, stderr=err, env=env,
        )
        compressor = subprocess.Popen(["zstd", "-q", "-T2", "-3", "-c"], stdin=tar.stdout, stdout=out, stderr=err)
        tar.stdout.close()
        compressed = compressor.wait()
        packed = tar.wait()
    if packed or compressed:
        raise RuntimeError(f"PACK_FAILED: {name}: {errors.read_text()[:1000]}")
    for entry in entries:
        s = (root / entry["path"]).lstat()
        if (s.st_size, s.st_mtime_ns, s.st_mode) != (entry["size"], entry["mtime_ns"], entry["mode"]):
            raise RuntimeError(f"SOURCE_CHANGED_DURING_PACK: {entry['path']}")
    log("packed", part=name, bytes=path.stat().st_size, entries=len(entries))
    receipt = upload_verified(client, bucket, path, f"{prefix}/parts/{name}")
    receipt["entries"] = len(entries)
    receipt["source_file_bytes"] = sum(e["size"] for e in entries if stat.S_ISREG(e["mode"]))
    atomic_json(receipt_path, receipt)
    path.unlink()
    names.unlink()
    errors.unlink()
    log("verified", **receipt)
    return receipt


def upload(args):
    client, endpoint = client_for(args.credentials)
    root = args.source.resolve()
    work = args.work.resolve()
    inventory_path = work / "inventory.json"
    if inventory_path.exists():
        entries = json.loads(inventory_path.read_text())
        if inventory(root) != entries:
            raise RuntimeError("SOURCE_CHANGED_SINCE_INVENTORY")
    else:
        entries = inventory(root)
        atomic_json(inventory_path, entries)
    batches = batches_of(entries)
    regular = [e for e in entries if stat.S_ISREG(e["mode"])]
    log("inventory", entries=len(entries), regular_files=len(regular), source_bytes=sum(e["size"] for e in regular), parts=len(batches))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(pack_upload, client, args.bucket, args.prefix, root, work, n, b) for n, b in enumerate(batches)]
        receipts = [f.result() for f in futures]
    if inventory(root) != entries:
        raise RuntimeError("SOURCE_CHANGED_BEFORE_COMPLETION")
    index_path = work / "files.jsonl"
    with index_path.open("w") as handle:
        for number, batch in enumerate(batches):
            for entry in batch:
                handle.write(json.dumps({**entry, "archive": f"parts/part-{number:04d}.tar.zst"}) + "\n")
    subprocess.run(["zstd", "-q", "-f", str(index_path), "-o", f"{index_path}.zst"], check=True)
    index_receipt = upload_verified(client, args.bucket, Path(f"{index_path}.zst"), f"{args.prefix}/files.jsonl.zst")
    reference = {
        "schema_version": 1,
        "status": "verified",
        "bucket": args.bucket,
        "prefix": args.prefix,
        "endpoint": endpoint,
        "source_repo": args.repo,
        "source_relative_path": args.relative_path,
        "source_absolute_path": str(root),
        "source_given_path": str(args.source.absolute()),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "regular_files": len(regular),
        "entries": len(entries),
        "source_file_bytes": sum(e["size"] for e in regular),
        "archive_bytes": sum(r["bytes"] for r in receipts),
        "verification": "Every archive: local SHA-256 and independently computed multipart MD5 ETag; R2 HEAD size, ETag and SHA-256 metadata matched. Source path/size/mtime/mode inventory unchanged before and after packing.",
        "file_index": index_receipt,
        "archives": receipts,
        "local_cleanup": None,
    }
    manifest_path = work / "manifest.json"
    atomic_json(manifest_path, reference)
    remote_manifest = upload_verified(client, args.bucket, manifest_path, f"{args.prefix}/manifest.json")
    if client.get_object(Bucket=args.bucket, Key=f"{args.prefix}/manifest.json")["Body"].read() != manifest_path.read_bytes():
        raise RuntimeError("MANIFEST_READBACK_FAILED")
    reference["archive_manifest"] = remote_manifest
    args.reference.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.reference, reference)
    log("complete", reference=str(args.reference), archive_bytes=reference["archive_bytes"], source_bytes=reference["source_file_bytes"], parts=len(receipts))


def load_reference(path):
    reference = json.loads(path.read_text())
    if reference["status"] != "verified":
        raise RuntimeError(f"ARCHIVE_NOT_VERIFIED: {path}")
    return reference


def verify_remote(client, reference):
    descriptors = [reference["file_index"], reference["archive_manifest"], *reference["archives"]]
    failed = [d["key"] for d in descriptors if not remote_matches(client, reference["bucket"], d)]
    if failed:
        raise RuntimeError(f"REMOTE_VERIFICATION_FAILED: {failed[:5]}")
    return len(descriptors)


def verify(args):
    client, _ = client_for(args.credentials)
    count = verify_remote(client, load_reference(args.reference))
    log("verified_remote", objects=count)


def download(client, bucket, descriptor, destination):
    client.download_file(bucket, descriptor["key"], str(destination))
    if destination.stat().st_size != descriptor["bytes"] or signatures(destination)[0] != descriptor["sha256"]:
        raise RuntimeError(f"ARCHIVE_CHECKSUM_MISMATCH: {descriptor['key']}")


def relocated(linkname, original_roots):
    for root in original_roots:
        try:
            return Path(linkname).relative_to(root)
        except ValueError:
            continue
    return None


def extract_archive(archive, destination, names, original_roots):
    with subprocess.Popen(["zstd", "-q", "-d", "-c", str(archive)], stdout=subprocess.PIPE) as process:
        try:
            with tarfile.open(fileobj=process.stdout, mode="r|") as contents:
                for member in contents:
                    if member.name not in names:
                        continue
                    if member.islnk() and member.linkname not in names:
                        raise RuntimeError(f"HARDLINK_TARGET_NOT_SELECTED: restore a broader prefix including {member.linkname}")
                    if member.issym() and Path(member.linkname).is_absolute():
                        target = relocated(member.linkname, original_roots)
                        if target is None:
                            link = destination / member.name
                            link.parent.mkdir(parents=True, exist_ok=True)
                            os.symlink(member.linkname, link)
                            continue
                        member = member.replace(linkname=os.path.relpath(destination / target, (destination / member.name).parent))
                    contents.extract(member, destination, filter="data")
        except BaseException:
            process.terminate()
            raise
        if process.wait() != 0:
            raise RuntimeError(f"DECOMPRESSION_FAILED: {archive}")


def restore(args):
    reference = load_reference(args.reference)
    client, _ = client_for(args.credentials)
    if not shutil.which("zstd"):
        raise RuntimeError("ZSTD_REQUIRED: install the zstd command before restoring")
    with tempfile.TemporaryDirectory(prefix="r2-restore-") as scratch:
        scratch = Path(scratch)
        compressed_index = scratch / "files.jsonl.zst"
        download(client, reference["bucket"], reference["file_index"], compressed_index)
        raw_index = subprocess.check_output(["zstd", "-q", "-d", "-c", str(compressed_index)], text=True)
        selected = [entry for line in raw_index.splitlines() if (entry := json.loads(line))["path"].startswith(args.prefix)]
        if not selected:
            raise RuntimeError(f"NO_MATCHING_PATHS: {args.prefix}")
        if args.list:
            for entry in selected:
                print(entry["path"])
            return
        if args.destination is None or args.destination.exists():
            raise SystemExit("--destination must name a new, nonexistent directory")
        names = {entry["path"] for entry in selected}
        parts = {entry["archive"] for entry in selected}
        required = sum(entry["size"] for entry in selected) + 3 * 1024**3
        args.destination.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(args.destination.parent).free < required:
            raise RuntimeError(f"INSUFFICIENT_SPACE: allow {required} bytes for restored files and a temporary archive")
        args.destination.mkdir()
        destination = args.destination.resolve()
        for descriptor in reference["archives"]:
            relative_key = descriptor["key"].removeprefix(reference["prefix"] + "/")
            if relative_key not in parts:
                continue
            archive = scratch / "part.tar.zst"
            download(client, reference["bucket"], descriptor, archive)
            roots = {reference["source_absolute_path"], reference.get("source_given_path") or reference["source_absolute_path"]}
            extract_archive(archive, destination, names, sorted(roots))
            archive.unlink()
            print(f"Restored verified archive: {relative_key}", flush=True)
        for entry in selected:
            path = destination / entry["path"]
            if not path.exists() and not path.is_symlink():
                raise RuntimeError(f"RESTORE_PATH_MISSING: {entry['path']}")
        print(f"Restored {len(selected)} paths into {destination}")


POINTER = """# Archived to Cloudflare R2

The contents of this directory were archived to R2 and removed locally on {deleted_at}.

- Bucket: {tick}{bucket}{tick} (private), prefix {tick}{prefix}/{tick}
- Endpoint: {endpoint}
- Reference manifest: {reference}
- {regular_files} files, {source_gib:.1f} GiB of file contents, stored as {parts} verified tar.zst parts

Credentials come from ~/.config/axport/r2.env on the owner's Mac; pass --credentials elsewhere.
List or restore selected paths into a new directory:

{fence}sh
uv run {tool} restore --reference {reference} --list --prefix SOME/PATH/
uv run {tool} restore --reference {reference} --prefix SOME/PATH/ --destination /path/to/new-dir
{fence}

New artifacts written here later are not archived automatically. Archive them the same way, into a new dated prefix:

{fence}sh
uv run {tool} upload --source {source} --bucket {bucket} --prefix {next_prefix} --reference NEW-REFERENCE.json
uv run {tool} cleanup --reference NEW-REFERENCE.json
{fence}
"""


def cleanup(args):
    reference = load_reference(args.reference)
    if reference.get("local_cleanup"):
        raise RuntimeError("ALREADY_CLEANED")
    client, _ = client_for(args.credentials)
    root = Path(reference["source_absolute_path"])
    expected = json.loads((args.work.resolve() / "inventory.json").read_text())
    current = inventory(root)
    if current != expected:
        archived = {e["path"]: e for e in expected}
        changed = [
            e["path"]
            for e in current
            if e["path"] not in archived or (not stat.S_ISDIR(e["mode"]) and archived[e["path"]] != e)
        ]
        if changed:
            raise RuntimeError(f"SOURCE_CHANGED_SINCE_ARCHIVE: re-run upload into a new prefix before cleanup: {changed[:5]}")
        log("resuming_interrupted_cleanup", remaining_entries=len(current))
    objects = verify_remote(client, reference)
    free_before = shutil.disk_usage(root).free
    for child in sorted(root.iterdir()):
        if child.is_dir() and not child.is_symlink():
            for directory, _, _ in os.walk(child):
                os.chmod(directory, os.lstat(directory).st_mode | stat.S_IRWXU)
            shutil.rmtree(child)
        else:
            child.unlink()
    reference["local_cleanup"] = {
        "deleted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": str(root),
        "removed_files_and_symlinks": sum(not stat.S_ISDIR(e["mode"]) for e in expected),
        "remote_objects_reverified": objects,
        "free_bytes_before": free_before,
        "free_bytes_after": shutil.disk_usage(root).free,
    }
    atomic_json(args.reference, reference)
    (root / "R2_ARCHIVE.md").write_text(
        POINTER.format(
            deleted_at=reference["local_cleanup"]["deleted_at"],
            bucket=reference["bucket"],
            prefix=reference["prefix"],
            endpoint=reference["endpoint"],
            reference=args.reference.resolve(),
            regular_files=reference["regular_files"],
            source_gib=reference["source_file_bytes"] / 1024**3,
            parts=len(reference["archives"]),
            tool=args.tool or Path(__file__).resolve(),
            source=root,
            next_prefix=reference["prefix"].rsplit("/", 1)[0] + "/YYYY-MM-DD",
            fence=FENCE,
            tick=TICK,
        )
    )
    log("cleaned", **reference["local_cleanup"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--credentials", type=Path, default=CREDENTIALS)
    sub = parser.add_subparsers(dest="command", required=True)
    up = sub.add_parser("upload")
    up.add_argument("--source", type=Path, required=True)
    up.add_argument("--bucket", required=True)
    up.add_argument("--prefix", required=True)
    up.add_argument("--reference", type=Path, required=True)
    up.add_argument("--work", type=Path)
    up.add_argument("--repo", default=None)
    up.add_argument("--relative-path", default=None)
    up.add_argument("--workers", type=int, default=2)
    ve = sub.add_parser("verify")
    ve.add_argument("--reference", type=Path, required=True)
    rs = sub.add_parser("restore")
    rs.add_argument("--reference", type=Path, required=True)
    rs.add_argument("--prefix", default="")
    rs.add_argument("--destination", type=Path)
    rs.add_argument("--list", action="store_true")
    cl = sub.add_parser("cleanup")
    cl.add_argument("--reference", type=Path, required=True)
    cl.add_argument("--work", type=Path)
    cl.add_argument("--tool", default=None, help="Tool path written into the pointer README")
    args = parser.parse_args()
    if args.command == "upload":
        args.work = args.work or Path.home() / ".cache/r2-archive" / args.bucket / args.prefix
        args.work.mkdir(parents=True, exist_ok=True)
        with (args.work / "upload.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            upload(args)
    elif args.command == "cleanup":
        reference = load_reference(args.reference)
        args.work = args.work or Path.home() / ".cache/r2-archive" / reference["bucket"] / reference["prefix"]
        cleanup(args)
    else:
        {"verify": verify, "restore": restore}[args.command](args)


if __name__ == "__main__":
    main()

import paramiko

HOST = "10.71.111.195"
USER = "orangepi"
PASSWORD = "orangepi"
ROOT = "/home/orangepi/Downloads/te"

TEXT_EXTS = {".py", ".sh", ".md", ".txt", ".log"}


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=15)
    sftp = client.open_sftp()

    def walk(remote_dir):
        for entry in sftp.listdir_attr(remote_dir):
            path = f"{remote_dir}/{entry.filename}"
            if entry.filename in ("__pycache__", ".idea"):
                continue
            if str(entry.longname).startswith("d"):
                yield from walk(path)
            else:
                yield path

    all_files = sorted(walk(ROOT))
    print(f"TOTAL_FILES={len(all_files)}")
    for path in all_files:
        rel = path.replace(ROOT + "/", "")
        ext = "." + rel.rsplit(".", 1)[-1] if "." in rel else ""
        if ext.lower() in TEXT_EXTS:
            try:
                with sftp.open(path, "r") as f:
                    data = f.read().decode("utf-8", errors="replace")
                print(f"\n{'='*60}\nFILE: {rel} ({len(data)} chars)\n{'='*60}")
                if len(data) > 8000:
                    print(data[:8000])
                    print(f"\n... [truncated, total {len(data)} chars] ...")
                else:
                    print(data)
            except Exception as e:
                print(f"\nFILE: {rel} READ_ERROR: {e}")
        else:
            size = sftp.stat(path).st_size
            print(f"BINARY: {rel} ({size} bytes)")

    sftp.close()
    client.close()


if __name__ == "__main__":
    main()

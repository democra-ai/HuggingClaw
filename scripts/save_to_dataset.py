import os
import tarfile
import tempfile
import sys
import time
from datetime import datetime

from huggingface_hub import HfApi

def main() -> None:
    """
    Backs up ~/.openclaw to Hugging Face Dataset with rolling history.
    Keeps the last 5 backups to prevent data loss from corruption.
    
    Env vars:
    - HF_TOKEN
    - OPENCLAW_DATASET_REPO
    """
    repo_id = os.environ.get("OPENCLAW_DATASET_REPO")
    token = os.environ.get("HF_TOKEN")

    state_dir = os.path.expanduser("~/.openclaw")

    if not repo_id or not token:
        print("[save_to_dataset] Missing configuration.", file=sys.stderr)
        return

    if not os.path.isdir(state_dir):
        print("[save_to_dataset] No state to save.", file=sys.stderr)
        return

    # 1. Validation: Ensure we have valid credentials before backing up
    wa_creds_dir = os.path.join(state_dir, "credentials", "whatsapp", "default")
    if os.path.isdir(wa_creds_dir):
        file_count = len([f for f in os.listdir(wa_creds_dir) if os.path.isfile(os.path.join(wa_creds_dir, f))])
        if file_count < 2:
             # Basic sanity check: needs at least creds.json + keys. 
             # Lowered from 10 to 2 to be less aggressive but still catch empty/broken states.
            print(f"[save_to_dataset] Skip: WhatsApp credentials incomplete ({file_count} files).", file=sys.stderr)
            return

    api = HfApi(token=token)
    
    # Sync system logs to state dir for persistence
    try:
        sys_log_path = "/home/node/logs"
        backup_log_path = os.path.join(state_dir, "logs/sys_logs")
        if os.path.exists(sys_log_path):
            if os.path.exists(backup_log_path):
                import shutil
                shutil.rmtree(backup_log_path)
            # Use shutil.copytree but ignore socket files if any
            import shutil
            shutil.copytree(sys_log_path, backup_log_path, ignore_dangling_symlinks=True)
            print(f"[save_to_dataset] Synced logs from {sys_log_path} to {backup_log_path}")
    except Exception as e:
        print(f"[save_to_dataset] Warning: Failed to sync logs: {e}")

    # Check for credentials
    creds_path = os.path.join(state_dir, "credentials/whatsapp/default/auth_info_multi.json")
    if os.path.exists(creds_path):
        print(f"[save_to_dataset] ✅ WhatsApp credentials found at {creds_path}")
    else:
        print("[save_to_dataset] ⚠️  WhatsApp credentials NOT found (user might need to login)")

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"state/backup-{timestamp}.tar.gz"

    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = os.path.join(tmpdir, "openclaw.tar.gz")

        try:
            with tarfile.open(tar_path, "w:gz") as tf:
                # Filter to exclude lock files or temp files if needed, but allow extensions
                def exclude_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
                    if info.name.endswith(".lock"):
                        return None
                    return info
                
                tf.add(state_dir, arcname=".", filter=exclude_filter)
        except Exception as e:
            print(f"[save_to_dataset] Failed to compress: {e}", file=sys.stderr)
            return

        print(f"[save_to_dataset] Uploading backup: {filename}")
        try:
            api.upload_file(
                path_or_fileobj=tar_path,
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="dataset",
            )
        except Exception as e:
            print(f"[save_to_dataset] Upload failed: {e}", file=sys.stderr)
            return

    # 2. Rotation: Delete old backups, keep last 5
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        # Match both .tar and .tar.gz for backward compatibility during transition
        backups = sorted([f for f in files if f.startswith("state/backup-") and (f.endswith(".tar") or f.endswith(".tar.gz"))])
        
        if len(backups) > 5:
            # Delete oldest
            to_delete = backups[:-5]
            print(f"[save_to_dataset] Rotating backups, deleting: {to_delete}")
            for old_backup in to_delete:
                api.delete_file(
                    path_in_repo=old_backup,
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=token
                )
    except Exception as e:
        print(f"[save_to_dataset] Rotation failed (non-fatal): {e}", file=sys.stderr)


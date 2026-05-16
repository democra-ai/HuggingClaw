#!/usr/bin/env python3
"""
Atomic Dataset Persistence for OpenClaw AI
Save state to Hugging Face Dataset with atomic operations
"""

import os
import sys
import json
import hashlib
import time
import tarfile
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import requests
import logging

from huggingface_hub import HfApi, CommitOperationAdd
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import hf_hub_download

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "module": "atomic-save", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

class AtomicDatasetSaver:
    """Atomic dataset persistence with proper error handling and retries"""
    
    def __init__(self, repo_id: str, dataset_path: str = "state"):
        self.repo_id = repo_id
        self.dataset_path = Path(dataset_path)
        self.api = HfApi()
        self.max_retries = 3
        self.base_delay = 1.0
        self.max_backups = 3
        
        logger.info("init", {
            "repo_id": repo_id,
            "dataset_path": dataset_path,
            "max_retries": self.max_retries,
            "max_backups": self.max_backups
        })
    
    def calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def create_backup(self, current_commit: Optional[str] = None) -> Optional[str]:
        """Create backup of current state before overwriting"""
        try:
            if not current_commit:
                return None
            
            # List current files in dataset
            files = self.api.list_repo_files(
                repo_id=self.repo_id,
                repo_type="dataset",
                revision=current_commit
            )
            
            # Only backup if there are existing state files
            state_files = [f for f in files if f.startswith(str(self.dataset_path))]
            if not state_files:
                return None
            
            # Create backup with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"backups/state_{timestamp}"
            
            logger.info("creating_backup", {
                "current_commit": current_commit,
                "backup_path": backup_path,
                "files_count": len(state_files)
            })
            
            # Download and create backup
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                
                # Download all state files
                for file_path in state_files:
                    file_content = hf_hub_download(
                        repo_id=self.repo_id,
                        repo_type="dataset",
                        filename=file_path,
                        revision=current_commit,
                        local_files_only=False
                    )
                    if file_content:
                        shutil.copy2(file_content, tmpdir_path / Path(file_path).name)
                
                # Create backup structure
                backup_files = []
                for file_path in state_files:
                    local_path = tmpdir_path / file_path
                    if local_path.exists():
                        backup_file_path = f"{backup_path}/{Path(file_path).name}"
                        backup_files.append(
                            CommitOperationAdd(
                                path_in_repo=backup_file_path,
                                path_or_fileobj=str(local_path)
                            )
                        )
                
                if backup_files:
                    # Commit backup
                    commit_info = self.api.create_commit(
                        repo_id=self.repo_id,
                        repo_type="dataset",
                        operations=backup_files,
                        commit_message=f"Backup state before update - {timestamp}",
                        parent_commit=current_commit
                    )
                    
                    logger.info("backup_created", {
                        "backup_commit": commit_info.oid,
                        "backup_path": backup_path
                    })
                    
                    return commit_info.oid
                
        except Exception as e:
            logger.error("backup_failed", {"error": str(e), "current_commit": current_commit})
            return None
    
    def cleanup_old_backups(self, current_commit: Optional[str] = None) -> None:
        """Clean up old backups, keeping only the most recent ones"""
        try:
            if not current_commit:
                return
            
            # List all files to find backups
            files = self.api.list_repo_files(
                repo_id=self.repo_id,
                repo_type="dataset",
                revision=current_commit
            )
            
            # Find backup directories
            backup_dirs = set()
            for file_path in files:
                if file_path.startswith("backups/state_"):
                    backup_dir = file_path.split("/")[1]  # Extract backup directory name
                    backup_dirs.add(backup_dir)
            
            # Keep only the most recent backups
            backup_list = sorted(backup_dirs)
            if len(backup_list) > self.max_backups:
                backups_to_remove = backup_list[:-self.max_backups]
                
                logger.info("cleaning_old_backups", {
                    "total_backups": len(backup_list),
                    "keeping": self.max_backups,
                    "removing": len(backups_to_remove),
                    "old_backups": backups_to_remove
                })
                
                # Note: In a real implementation, we would delete these files
                # For now, we just log what would be cleaned up
                
        except Exception as e:
            logger.error("backup_cleanup_failed", {"error": str(e)})
    
    def save_state_atomic(self, state_data: Dict[str, Any], source_paths: List[str]) -> Dict[str, Any]:
        """
        Save state to dataset atomically
        
        Args:
            state_data: Dictionary containing state information
            source_paths: List of file paths to include in the state
            
        Returns:
            Dictionary with operation result
        """
        operation_id = f"save_{int(time.time())}"
        
        logger.info("starting_atomic_save", {
            "operation_id": operation_id,
            "state_keys": list(state_data.keys()),
            "source_paths": source_paths
        })
        
        try:
            # Get current commit to use as parent
            try:
                repo_info = self.api.repo_info(
                    repo_id=self.repo_id,
                    repo_type="dataset"
                )
                current_commit = repo_info.sha
                logger.info("current_commit_found", {"commit": current_commit})
            except RepositoryNotFoundError:
                current_commit = None
                logger.info("repository_not_found", {"action": "creating_new_repo"})
            
            # Create backup before making changes
            backup_commit = self.create_backup(current_commit)
            
            # Create temporary directory for state files
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                state_dir = tmpdir_path / self.dataset_path
                state_dir.mkdir(parents=True, exist_ok=True)
                
                # Save state metadata
                metadata = {
                    "timestamp": datetime.now().isoformat(),
                    "operation_id": operation_id,
                    "checksum": None,
                    "backup_commit": backup_commit,
                    "state_data": state_data
                }
                
                metadata_path = state_dir / "metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                
                # Copy source files to state directory
                operations = [CommitOperationAdd(path_in_repo="state/metadata.json", path_or_fileobj=str(metadata_path))]
                
                for source_path in source_paths:
                    source = Path(source_path)
                    if source.exists():
                        dest_path = state_dir / source.name
                        shutil.copy2(source, dest_path)
                        
                        # Calculate checksum for integrity
                        checksum = self.calculate_checksum(dest_path)
                        
                        operations.append(
                            CommitOperationAdd(
                                path_in_repo=f"state/{source.name}",
                                path_or_fileobj=str(dest_path)
                            )
                        )
                        
                        logger.info("file_added", {
                            "source": source_path,
                            "checksum": checksum,
                            "operation_id": operation_id
                        })
                
                # Create final metadata with checksums
                final_metadata = metadata.copy()
                final_metadata["checksum"] = hashlib.sha256(
                    json.dumps(state_data, sort_keys=True).encode()
                ).hexdigest()
                
                # Update metadata file
                with open(metadata_path, "w") as f:
                    json.dump(final_metadata, f, indent=2)
                
                # Atomic commit to dataset
                commit_info = self.api.create_commit(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    operations=operations,
                    commit_message=f"Atomic state update - {operation_id}",
                    parent_commit=current_commit
                )
                
                # Clean up old backups
                self.cleanup_old_backups(commit_info.oid)
                
                result = {
                    "success": True,
                    "operation_id": operation_id,
                    "commit_id": commit_info.oid,
                    "backup_commit": backup_commit,
                    "timestamp": datetime.now().isoformat(),
                    "files_count": len(source_paths)
                }
                
                logger.info("atomic_save_completed", result)
                return result
                
        except Exception as e:
            error_result = {
                "success": False,
                "operation_id": operation_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.error("atomic_save_failed", error_result)
            raise Exception(f"Atomic save failed: {str(e)}")

def main():
    """Main function for command line usage"""
    if len(sys.argv) < 3:
        print(json.dumps({
            "error": "Usage: python save_to_dataset_atomic.py <repo_id> <source_path1> [source_path2...]",
            "status": "error"
        }, indent=2))
        sys.exit(1)
    
    repo_id = sys.argv[1]
    source_paths = sys.argv[2:]
    
    # Validate source paths
    for path in source_paths:
        if not os.path.exists(path):
            print(json.dumps({
                "error": f"Source path does not exist: {path}",
                "status": "error"
            }, indent=2))
            sys.exit(1)
    
    try:
        # Create state data (can be enhanced to read from environment or config)
        state_data = {
            "environment": "production",
            "version": "1.0.0",
            "platform": "huggingface-spaces",
            "timestamp": datetime.now().isoformat()
        }
        
        saver = AtomicDatasetSaver(repo_id)
        result = saver.save_state_atomic(state_data, source_paths)
        
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(json.dumps({
            "error": str(e),
            "status": "error"
        }, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    main()
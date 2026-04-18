"""Profile management (System Zarządzania Profilem): SQLite-backed persistence + service."""

from dietary_advisor.profile_manager.service import ProfileService
from dietary_advisor.profile_manager.store import ProfileStore

__all__ = ["ProfileService", "ProfileStore"]

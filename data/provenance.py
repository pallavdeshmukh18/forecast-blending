"""
Data provenance tracking and metadata standardization for real and synthetic weather datasets.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DataProvenance(BaseModel):
    """
    Standard metadata record for every dataset entering the blending system.
    """
    source: str = Field(description="Origin source name, e.g. 'ECMWF_IFS', 'NOAA_GFS', 'IMD_AWS'")
    dataset_name: str = Field(description="Identifier for the dataset or run")
    provider: str = Field(description="Organization or data provider, e.g. 'ECMWF', 'NOAA', 'SyntheticEngine'")
    variable: str = Field(description="Standardized meteorological variable name")
    units: str = Field(description="Physical units after normalization, e.g. 'celsius', 'mm', 'm/s'")
    spatial_resolution: Optional[str] = Field(default="0.25 deg", description="Spatial resolution")
    temporal_resolution: Optional[str] = Field(default="6h", description="Temporal step resolution")
    initialization_time: Optional[str] = Field(default=None, description="Model cycle initialization time")
    valid_time: Optional[str] = Field(default=None, description="Forecast valid timestamp")
    download_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Timestamp when dataset was ingested or generated"
    )
    license_info: str = Field(
        default="Open Meteorological Data / Research License",
        description="Data license or source copyright terms"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional custom metadata tags")


class ProvenanceTracker:
    """
    Central repository tracker for dataset provenance manifests.
    """
    def __init__(self, storage_path: Optional[str | Path] = "reports/data_provenance.json"):
        self.storage_path = Path(storage_path) if storage_path else Path("reports/data_provenance.json")
        self.records: List[DataProvenance] = []
        self._load_existing()

    def _load_existing(self) -> None:
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.records = [DataProvenance(**item) for item in data]
            except Exception:
                self.records = []

    def add_provenance(self, provenance: DataProvenance) -> None:
        """Add a provenance record and save the manifest."""
        self.records.append(provenance)
        self.save()

    def register_dataset(
        self,
        source: str,
        dataset_name: str,
        provider: str,
        variable: str,
        units: str,
        spatial_resolution: Optional[str] = "0.25 deg",
        temporal_resolution: Optional[str] = "6h",
        initialization_time: Optional[str] = None,
        valid_time: Optional[str] = None,
        license_info: str = "Research / Open Data License",
        **kwargs: Any
    ) -> DataProvenance:
        """Create and register a new provenance record."""
        prov = DataProvenance(
            source=source,
            dataset_name=dataset_name,
            provider=provider,
            variable=variable,
            units=units,
            spatial_resolution=spatial_resolution,
            temporal_resolution=temporal_resolution,
            initialization_time=initialization_time,
            valid_time=valid_time,
            license_info=license_info,
            metadata=kwargs
        )
        self.add_provenance(prov)
        return prov

    def save(self) -> None:
        """Persist provenance log to JSON."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump([r.model_dump() for r in self.records], f, indent=2)

    def get_summary(self) -> Dict[str, Any]:
        """Return high-level summary of recorded datasets."""
        return {
            "total_datasets": len(self.records),
            "providers": list({r.provider for r in self.records}),
            "variables": list({r.variable for r in self.records}),
            "sources": list({r.source for r in self.records}),
        }


# Global singleton tracker
_PROVENANCE_TRACKER = None

def get_provenance_tracker(storage_path: Optional[str | Path] = None) -> ProvenanceTracker:
    global _PROVENANCE_TRACKER
    if _PROVENANCE_TRACKER is None or storage_path is not None:
        _PROVENANCE_TRACKER = ProvenanceTracker(storage_path or "reports/data_provenance.json")
    return _PROVENANCE_TRACKER

from typing import List
from pydantic import BaseSettings, Field
from pathlib import Path


class GCPSettings(BaseSettings):
    project_id: str = Field("grift-forex", description="GCP project ID")
    region: str = Field("us-east1", description="GCP region")


class OandaSettings(BaseSettings):
    environment: str = Field("practice", description="OANDA environment (practice/live)")
    currency_pairs: List[str] = Field(
        default=[
            "EUR_USD", "GBP_USD", "USD_JPY", "USD_CAD",
            "AUD_USD", "NZD_USD", "USD_CHF", "EUR_GBP"
        ],
        description="Currency pairs to monitor"
    )


class PubSubSettings(BaseSettings):
    topic_raw: str = Field("w.latent.raw", description="Raw latent vectors")
    topic_rollup_5s: str = Field("w.latent.5s", description="5-second rollup")
    topic_rollup_15s: str = Field("w.latent.15s", description="15-second rollup")
    topic_matrix_5s: str = Field("M.latent.5s", description="5-second transition matrices")
    topic_matrix_15s: str = Field("M.latent.15s", description="15-second transition matrices")


class FirestoreSettings(BaseSettings):
    collection_checkpoints: str = Field("tick_checkpoints", description="Checkpoint collection")
    checkpoint_batch_size: int = Field(100, description="Number of ticks per checkpoint batch")


class Settings(BaseSettings):
    # Nested configuration sections
    gcp: GCPSettings = GCPSettings()
    oanda: OandaSettings = OandaSettings()
    pubsub: PubSubSettings = PubSubSettings()
    firestore: FirestoreSettings = FirestoreSettings()

    # Environment and debug settings
    environment: str = Field("development", description="Current environment (development/production)")
    debug: bool = Field(False, description="Enable debug mode")

    class Config:
        env_file = ".env.local"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Create a singleton instance of the settings
settings = Settings()

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "practice_account_id" {
  description = "OANDA practice account ID"
  type        = string
  sensitive   = true
}

variable "practice_api_token" {
  description = "OANDA practice API token"
  type        = string
  sensitive   = true
}

resource "google_secret_manager_secret" "oanda_practice_account" {
  project   = var.project_id
  secret_id = "oanda-practice-account-id"

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "oanda_practice_account_version" {
  secret      = google_secret_manager_secret.oanda_practice_account.id
  secret_data = var.practice_account_id
}

resource "google_secret_manager_secret" "oanda_practice_token" {
  project   = var.project_id
  secret_id = "oanda-practice-api-token"

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "oanda_practice_token_version" {
  secret      = google_secret_manager_secret.oanda_practice_token.id
  secret_data = var.practice_api_token
}

# IAM bindings for the GKE service account to access secrets
resource "google_secret_manager_secret_iam_member" "gke_practice_account" {
  project   = var.project_id
  secret_id = google_secret_manager_secret.oanda_practice_account.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${var.gke_service_account}"
}

resource "google_secret_manager_secret_iam_member" "gke_practice_token" {
  project   = var.project_id
  secret_id = google_secret_manager_secret.oanda_practice_token.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${var.gke_service_account}"
}

variable "gke_service_account" {
  description = "Service account email for GKE workloads"
  type        = string
}

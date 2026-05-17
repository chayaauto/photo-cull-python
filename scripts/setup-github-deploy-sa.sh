#!/usr/bin/env bash
# One-time: create GCP service account for GitHub Actions deploy.
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-photo-cull-2026}"
SA_NAME="${SA_NAME:-github-deploy}"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
KEY_FILE="${KEY_FILE:-$(dirname "$0")/../github-sa-key.json}"

echo "Project: ${PROJECT_ID}"
echo "Service account: ${SA_EMAIL}"

gcloud config set project "${PROJECT_ID}"

if ! gcloud iam service-accounts describe "${SA_EMAIL}" &>/dev/null; then
  gcloud iam service-accounts create "${SA_NAME}" \
    --display-name="GitHub Actions Cloud Run deploy"
fi

for role in \
  roles/run.admin \
  roles/iam.serviceAccountUser \
  roles/cloudbuild.builds.editor \
  roles/artifactregistry.admin \
  roles/storage.admin; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="${role}" \
    --quiet >/dev/null
  echo "Granted ${role}"
done

gcloud iam service-accounts keys create "${KEY_FILE}" \
  --iam-account="${SA_EMAIL}"

echo ""
echo "Done. Key saved to: ${KEY_FILE}"
echo ""
echo "Next — add GitHub secret:"
echo "  1. Open https://github.com/chayaauto/photo-cull-python/settings/secrets/actions"
echo "  2. New repository secret"
echo "  3. Name: GCP_SA_KEY"
echo "  4. Value: paste entire contents of ${KEY_FILE}"
echo ""
echo "Then push to main — deploy runs automatically."

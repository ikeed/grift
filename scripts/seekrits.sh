#!/bin/bash

# This script creates Google Cloud Secrets for OANDA account credentials.
# Make sure you have the Google Cloud SDK installed and authenticated.

#You can create an oanda account here: https://www.oanda.com/us-en/trading/accounts/practice-account/

# Set your OANDA practice (fake money) account credentials here
PRACTICE_ACCOUNT_ID="YOUR_PRACTICE_ACCOUNT_ID"
PRACTICE_ACCESS_TOKEN="YOUR_PRACTICE_ACCESS_TOKEN"

# If you have a production (real money) account, uncomment the following lines and set the values
#PRODUCTION_ACCOUNT_ID=""
#PRODUCTION_ACCESS_TOKEN=""

# For practice environment
gcloud secrets create oanda-practice-account-id --replication-policy="automatic"
echo -n "$PRACTICE_ACCOUNT_ID" | gcloud secrets versions add oanda-practice-account-id --data-file=-

gcloud secrets create oanda-practice-api-token --replication-policy="automatic"
echo -n "$PRACTICE_ACCESS_TOKEN" | gcloud secrets versions add oanda-practice-api-token --data-file=-

# For production environment
#gcloud secrets create oanda-production-account-id --replication-policy="automatic"
if [[ -n "$PRODUCTION_ACCOUNT_ID" && -n "$PRODUCTION_ACCESS_TOKEN" ]]; then
  echo -n "$PRODUCTION_ACCOUNT_ID" | gcloud secrets versions add oanda-production-account-id --data-file=-
  #gcloud secrets create oanda-production-api-token --replication-policy="automatic"
  echo -n "$PRODUCTION_ACCESS_TOKEN" | gcloud secrets versions add oanda-production-api-token --data-file=-
fi
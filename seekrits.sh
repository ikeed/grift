# For practice environment
gcloud secrets create oanda-practice-account-id --replication-policy="automatic"
echo -n "YOUR_PRACTICE_ACCOUNT_ID" | gcloud secrets versions add oanda-practice-account-id --data-file=-

gcloud secrets create oanda-practice-api-token --replication-policy="automatic"
echo -n "YOUR_PRACTICE_ACCESS_TOKEN" | gcloud secrets versions add oanda-practice-api-token --data-file=-

# For production environment
#gcloud secrets create oanda-production-account-id --replication-policy="automatic"
echo -n "YOUR_PRODUCTION_ACCOUNT_ID" | gcloud secrets versions add oanda-production-account-id --data-file=-

#gcloud secrets create oanda-production-api-token --replication-policy="automatic"
echo -n "YOUR_PRODUCTION_ACCESS_TOKEN" | gcloud secrets versions add oanda-production-api-token --data-file=-

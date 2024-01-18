inherit from Maestro and override `create_generator` and/or `handle_item`

secrets:
- `DATABASE_URL`
- `SUPABASE_API_KEY` 

options:
- `MODEL`
- `SUPABASE_URL`
- `ADMIN_URL`

flags:
- `FREE`
- `EXIT`
- `EXIT_ON_LOAD`
- `POWEROFF`
- `TWITTER`: post tweets, requires comma-seperated TWITTER_CREDS and TwitterAPI to be installed

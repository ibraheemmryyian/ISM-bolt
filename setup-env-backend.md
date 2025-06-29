# Backend Environment Setup

## Required Environment Variables

Create a `.env` file in the `backend/` directory with the following variables:

```env
# Server Configuration
PORT=5000
NODE_ENV=development

# Frontend URL for CORS
FRONTEND_URL=http://localhost:5173

# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key

# Security
JWT_SECRET=your_jwt_secret_here_change_this_in_production
SESSION_SECRET=your_session_secret_here_change_this_in_production

# py AI Engine Path
py_PATH=py

# Logging
LOG_LEVEL=info

# Rate Limiting
RATE_LIMIT_WINDOW_MS=900000
RATE_LIMIT_MAX_REQUESTS=100
```

## Steps to Set Up:

1. **Create the .env file:**
   ```bash
   cd backend
   copy env.example .env
   ```

2. **Update the Supabase credentials:**
   - Replace `https://your-project.supabase.co` with your actual Supabase project URL
   - Replace `your-anon-key` with your actual Supabase anon key

3. **Restart the backend server:**
   ```bash
   npm start
   ```

## Testing the Setup:

After creating the .env file, you should see:
- ✅ Supabase connection successful
- No more "SUPABASE_URL not configured" errors
- AI onboarding should work and save materials to the database

## Troubleshooting:

If you still see connection errors:
1. Verify your Supabase credentials are correct
2. Check that the materials table exists in your Supabase database
3. Ensure the backend server is running on port 5000
4. Check the browser console for any CORS errors 
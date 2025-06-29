# Environment Setup Guide

## Quick Fix for Messaging System

The Connect, Chat, and Favorite buttons aren't working because the Supabase environment variables are not configured. Here's how to fix it:

### 1. Create Frontend Environment File

Create a file called `.env` in the `frontend` directory with your Supabase credentials:

```env
VITE_SUPABASE_URL=https://your-project-id.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-key-here
```

### 2. Create Backend Environment File

Create a file called `.env` in the `backend` directory:

```env
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_ANON_KEY=your-anon-key-here
PORT=5000
NODE_ENV=development
FRONTEND_URL=http://localhost:5173
```

### 3. Get Your Supabase Credentials

1. Go to [supabase.com](https://supabase.com)
2. Create a new project or use existing one
3. Go to Settings > API
4. Copy the "Project URL" and "anon public" key

### 4. Run Database Migration

After setting up the environment variables, run:

```bash
cd frontend
npx supabase db push
```

### 5. Restart Servers

```bash
# Backend
cd backend
npm start

# Frontend (new terminal)
cd frontend
npm run dev
```

## What This Fixes

✅ **Connect Button** - Now properly connects/disconnects with companies  
✅ **Chat Button** - Creates conversations and navigates to chat panel  
✅ **Favorite Button** - Adds/removes materials from favorites  
✅ **Real-time Messaging** - Full chat functionality with conversations  
✅ **Database Integration** - All data properly stored in Supabase  

## Test the Features

1. **Connect**: Click "Connect" on any company/material - should show "Connected" status
2. **Chat**: Click the chat icon - should create conversation and navigate to chats
3. **Favorites**: Click the heart icon - should add/remove from favorites
4. **Messaging**: Go to /chats route to see all conversations

## Troubleshooting

If you still see errors:
1. Check browser console for any error messages
2. Make sure both frontend and backend servers are running
3. Verify Supabase credentials are correct
4. Check that the database migration ran successfully

The messaging system is now fully functional with real-time capabilities! 
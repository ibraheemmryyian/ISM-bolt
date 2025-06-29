@echo off
echo Creating environment files with your Supabase credentials...

echo Creating frontend/.env file...
(
echo VITE_SUPABASE_URL=https://jifkiwbxnttrkdrdcose.supabase.co
echo VITE_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImppZmtpd2J4bnR0cmtkcmRjb3NlIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDMwMTY3ODcsImV4cCI6MjA1ODU5Mjc4N30.S46lWR5O724iTs7CCPQb_cmEPzfOVWqFvWQk_pW7Zvk
) > frontend\.env

echo Creating backend/.env file...
(
echo SUPABASE_URL=https://jifkiwbxnttrkdrdcose.supabase.co
echo SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImppZmtpd2J4bnR0cmtkcmRjb3NlIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDMwMTY3ODcsImV4cCI6MjA1ODU5Mjc4N30.S46lWR5O724iTs7CCPQb_cmEPzfOVWqFvWQk_pW7Zvk
echo PORT=5000
echo NODE_ENV=development
echo FRONTEND_URL=http://localhost:5173
) > backend\.env

echo Environment files created successfully!
echo.
echo Next steps:
echo 1. Run the database migration: cd frontend ^&^& npx supabase db push
echo 2. Restart the backend server: cd backend ^&^& npm start
echo 3. Restart the frontend server: cd frontend ^&^& npm run dev
echo.
pause 
# SymbioFlows Demo System

This README provides instructions on how to set up and run the SymbioFlows demo system for creating a video demonstration of the complete user journey.

## Overview

The demo system allows you to showcase:
1. Account creation
2. AI onboarding process
3. Material listings generation
4. Matches discovery
5. Dashboard features

## Prerequisites

- Python 3.7+
- Node.js 14+
- npm 6+
- PostgreSQL (optional, for full database functionality)
- Supabase account (optional, for authentication)

## Quick Start

### For Linux/Mac Users:

1. Make the script executable:
   ```bash
   chmod +x scripts/run_demo.sh
   ```

2. Run the demo system:
   ```bash
   ./scripts/run_demo.sh
   ```

### For Windows Users:

1. Run the demo system:
   ```
   scripts\run_demo.bat
   ```

## Manual Setup

If you prefer to set up each component manually:

1. Prepare the demo data:
   ```bash
   python scripts/demo_video_preparation.py
   ```

2. Start the backend:
   ```bash
   cd backend
   npm start
   ```

3. In a separate terminal, start the frontend:
   ```bash
   cd frontend
   npm run dev
   ```

4. Access the application at http://localhost:3000

## Demo Guide

Follow the step-by-step guide in `docs/DEMO_VIDEO_GUIDE.md` to create a compelling demo video.

## Configuration

### Environment Variables

You can configure the demo by setting these environment variables:

- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_KEY`: Your Supabase API key
- `DB_HOST`: PostgreSQL host (default: localhost)
- `DB_PORT`: PostgreSQL port (default: 5432)
- `DB_NAME`: PostgreSQL database name (default: postgres)
- `DB_USER`: PostgreSQL username (default: postgres)
- `DB_PASSWORD`: PostgreSQL password (default: postgres)

### Sample Data

The demo preparation script creates sample data in the `data/` directory:
- `sample_companies.csv`: Company information
- `sample_waste_streams.csv`: Waste stream data

You can replace these files with your own data before running the demo preparation script.

## Troubleshooting

### Database Connection Issues

If you encounter database connection issues:
1. Check that PostgreSQL is running
2. Verify your database credentials
3. Ensure the required tables exist

### AI Service Issues

If the AI onboarding service is not working:
1. Check that the backend is running
2. Look for errors in the backend console
3. Ensure the AI service dependencies are installed

### Frontend Issues

If the frontend is not displaying properly:
1. Clear your browser cache
2. Check for JavaScript errors in the browser console
3. Ensure the frontend is running on the correct port

## Support

For additional support, contact the development team or open an issue on the project repository.
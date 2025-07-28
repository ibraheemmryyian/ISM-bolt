# Demo Video Guide: Complete User Journey

This guide outlines the steps to create a compelling demo video showcasing the complete user journey from account creation through AI onboarding to viewing material listings and matches.

## Prerequisites

1. Run the demo preparation script:
   ```bash
   python scripts/demo_video_preparation.py
   ```
   This will:
   - Set up a demo account
   - Import real company data with waste streams
   - Generate material matches

2. Ensure the frontend and backend are running:
   ```bash
   # In one terminal
   cd backend
   npm start
   
   # In another terminal
   cd frontend
   npm run dev
   ```

## Demo Video Script

### 1. Introduction (30 seconds)
- Introduce the platform: "Today I'll demonstrate our industrial symbiosis platform that helps companies find partners for their waste materials"
- Mention the key features: "You'll see how our AI onboarding process identifies material opportunities and matches companies together"

### 2. Account Creation (1 minute)
- Navigate to the signup page
- Fill in the demo account details provided by the preparation script
- Complete the basic account setup
- Show the welcome screen

### 3. AI Onboarding Process (2-3 minutes)
- Click on "Start AI Onboarding" from the dashboard
- Fill in the company information:
  - Industry: Choose an industry with significant waste streams (e.g., Manufacturing, Food & Beverage)
  - Products: Describe the company's products in detail
  - Production Volume: Provide specific quantities (e.g., "500 tons per month")
  - Processes: Describe the manufacturing processes that generate waste

- Highlight the AI confidence meter increasing as you add more detailed information
- Point out how the system requires 95% confidence before proceeding
- Submit the onboarding form
- Show the AI analysis in progress

### 4. Review Material Listings (2 minutes)
- When the AI analysis completes, review the generated material listings
- Point out:
  - The AI has identified both waste materials and requirements
  - Each listing includes detailed information about quantity, unit, and description
  - The confidence score for each listing
  - The option to edit or add new listings
- Make a small edit to one of the listings to demonstrate the functionality
- Add a new listing to show how users can supplement the AI-generated content
- Save the listings

### 5. Dashboard Overview (1-2 minutes)
- Explore the main dashboard
- Highlight key metrics:
  - Total savings
  - Carbon reduced
  - Partnerships formed
  - Sustainability score
- Show the company profile section
- Point out the AI-generated materials section
- Demonstrate the AI-powered recommendations

### 6. Viewing Matches (2 minutes)
- Navigate to the matches section
- Show how the system has identified potential matches for the company's materials
- Highlight match scores and compatibility metrics
- Explain how the matching algorithm works
- Click on a match to view detailed information
- Show the potential environmental and financial benefits

### 7. Taking Action (1 minute)
- Demonstrate how to initiate contact with a potential match
- Show the messaging interface
- Explain the next steps in the partnership process

### 8. Conclusion (30 seconds)
- Summarize the key benefits demonstrated:
  - Efficient AI onboarding
  - Accurate material listings
  - Intelligent matching
  - Environmental and financial impact
- Mention any upcoming features or improvements

## Recording Tips

1. **Preparation**:
   - Have all credentials ready
   - Clear browser cache before starting
   - Use a high-resolution screen recording tool
   - Test microphone quality

2. **Presentation**:
   - Speak clearly and at a moderate pace
   - Highlight key features as you demonstrate them
   - Explain what you're doing at each step
   - Emphasize the value proposition throughout

3. **Technical Setup**:
   - Record at 1080p or higher resolution
   - Use a second monitor for script/notes if possible
   - Ensure stable internet connection
   - Close unnecessary applications and notifications

4. **Post-Production**:
   - Add captions or subtitles
   - Include transitions between major sections
   - Add logo/branding elements
   - Consider adding background music

## Troubleshooting

If you encounter issues during the demo recording:

1. **Database Connection Issues**:
   - Check that the backend services are running
   - Verify database credentials in .env files

2. **AI Onboarding Not Processing**:
   - Ensure the AI service is running
   - Check backend logs for errors

3. **No Matches Appearing**:
   - Run the demo preparation script again
   - Check that materials have been properly imported

4. **Frontend Display Issues**:
   - Clear browser cache
   - Try a different browser
   - Check console for JavaScript errors

For additional support, contact the development team.
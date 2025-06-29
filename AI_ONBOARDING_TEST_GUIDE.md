# 🤖 AI Onboarding Test Guide

## 🎯 Overview
This guide helps you test the AI onboarding functionality to ensure everything is working correctly.

## 🚀 Quick Start

### 1. Start the Backend
```bash
cd backend
npm start
```

### 2. Start the Frontend
```bash
cd frontend
npm run dev
```

### 3. Access the AI Onboarding
Navigate to: `http://localhost:5173/onboarding`

## 🧪 Testing Checklist

### ✅ UI Components Test
- [ ] Page loads without errors
- [ ] AI assistant greeting appears
- [ ] Input field is visible and focused
- [ ] Send button is enabled/disabled correctly
- [ ] Progress bar shows current step
- [ ] Sidebar shows company info

### ✅ Data Input Test
- [ ] Can type in company name
- [ ] Can select industry from dropdown
- [ ] Can enter location
- [ ] Can select employee count
- [ ] Can describe products
- [ ] Can list materials
- [ ] Can enter production volume
- [ ] Can describe processes
- [ ] Can select sustainability goals

### ✅ AI Interaction Test
- [ ] AI responds to each input
- [ ] Conversation history is maintained
- [ ] Progress updates correctly
- [ ] Company data is saved
- [ ] Next question appears automatically

### ✅ Progress Tracking Test
- [ ] Progress bar updates
- [ ] Step counter increases
- [ ] Completed steps are marked
- [ ] Current step is highlighted

### ✅ Completion Test
- [ ] All steps can be completed
- [ ] Completion message appears
- [ ] Next steps options are shown
- [ ] Can navigate to dashboard/marketplace

## 🔧 Troubleshooting

### Common Issues

#### 1. Input Field Not Working
**Problem**: Can't type in input fields
**Solution**: 
- Check if Input component exists: `frontend/src/components/ui/input.tsx`
- Verify no JavaScript errors in browser console
- Ensure input field is not disabled

#### 2. AI Not Responding
**Problem**: AI doesn't respond to inputs
**Solution**:
- Check backend is running on port 3001
- Verify API endpoints are working
- Check browser network tab for errors

#### 3. Progress Not Updating
**Problem**: Progress bar doesn't move
**Solution**:
- Check if Progress component exists: `frontend/src/components/ui/progress.tsx`
- Verify state updates are working
- Check for React rendering issues

#### 4. Minimize/Maximize Not Working
**Problem**: Can't minimize or maximize the onboarding
**Solution**:
- Check if Button component supports size prop
- Verify Minimize2/Maximize2 icons are imported
- Check CSS classes are applied correctly

## 📊 Expected Behavior

### First-Time User Experience
1. **Large Display**: Full-screen onboarding experience
2. **Guided Flow**: Step-by-step questions with AI assistance
3. **Progress Visibility**: Clear progress indication
4. **Data Collection**: Comprehensive company profile creation

### Returning User Experience
1. **Minimized Mode**: Compact floating assistant
2. **Quick Access**: Easy to expand when needed
3. **Progress Resume**: Can continue from where left off
4. **Data Persistence**: Previous inputs are saved

## 🎨 UI Components Required

Make sure these components exist in `frontend/src/components/ui/`:
- `card.tsx` - Card, CardContent, CardHeader, CardTitle
- `button.tsx` - Button with variant and size props
- `input.tsx` - Input field component
- `textarea.tsx` - Textarea component
- `badge.tsx` - Badge with variant prop
- `progress.tsx` - Progress bar component

## 🔍 Debug Mode

To enable debug mode, add this to your browser console:
```javascript
localStorage.setItem('debug-onboarding', 'true');
```

This will show additional logging information.

## 📈 Performance Metrics

Monitor these metrics during testing:
- **Response Time**: AI responses should be < 2 seconds
- **Input Lag**: Typing should be responsive
- **Memory Usage**: Should not increase significantly
- **Error Rate**: Should be 0% for successful flows

## 🚨 Error Reporting

If you encounter errors:
1. Check browser console for JavaScript errors
2. Check network tab for failed requests
3. Check backend logs for server errors
4. Report issues with specific steps to reproduce

## ✅ Success Criteria

The AI onboarding is working correctly when:
- [ ] All UI components render without errors
- [ ] Users can input all required data
- [ ] AI responds appropriately to each input
- [ ] Progress is tracked accurately
- [ ] Completion flow works end-to-end
- [ ] Minimize/maximize functionality works
- [ ] Data persists between sessions
- [ ] Performance is acceptable (< 2s response time)

## 🎯 Next Steps After Testing

Once testing is complete:
1. **Fix any issues** found during testing
2. **Optimize performance** if needed
3. **Add error handling** for edge cases
4. **Improve UX** based on feedback
5. **Deploy to production** when ready 
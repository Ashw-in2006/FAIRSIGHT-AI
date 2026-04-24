# FairSight AI

Local bias-audit app with a FastAPI backend and React frontend.

## Run Locally

1. Install Node.js and Python.
2. Option A: Run `npm start` from the project root.
3. Option B: Double-click `run-local.bat`.
4. Option C: Open two terminals and run:
   - `cd backend && start.bat`
   - `cd frontend && start.bat`
5. Open the frontend in your browser and upload a CSV.

## Notes

- Backend runs on `http://localhost:8001`
- Frontend runs on `http://localhost:3000`
- Default model is Random Forest for better accuracy.
- Gemini is optional. If `GEMINI_API_KEY` is not set, the app uses local fallback explanations.
- If the frontend dependency folder gets corrupted, run `npm run reset-frontend` from the root.
- For Netlify/Vercel, set `REACT_APP_API_URL` to your deployed backend URL.

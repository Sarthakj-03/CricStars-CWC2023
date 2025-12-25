# CricStars-CWC2023
CricStars is a comprehensive platform offering detailed statistics and insights on player performances in the Cricket World Cup 2023. 
This is a simple project I made to practice Python and Streamlit.  
It shows some basic stats from ICC Cricket World Cup 2023 like batting, bowling and fielding.
Tech stack
- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
What this app does
- Shows overview of the dataset (rows, teams, matches).
- Shows top run scorers and team wise batting stats.
- Shows top wicket takers and team wise bowling stats.
- Simple fielding stats like dismissals (if available in csv).
- Player explorer page where you can select a player and see his batting and bowling numbers.
I mainly built it to learn how to:
- Load a csv with Pandas.
- Do basic data analysis.
- Make charts with Matplotlib / Seaborn.
- Create an interactive dashboard with Streamlit.
Dataset
You have to download any ICC Cricket World Cup 2023 dataset (ODI) from the internet (for example from Kaggle) and save it as: odi_wc_2023.csv

Put the csv file in the same folder as this project.  
If your column names are different, you may have to edit the helper functions in `cricstars_app.py`:

- `get_batting_view()`
- `get_bowling_view()`

Inside those functions I try to guess columns like `batting_team`, `player`, `runs`, `wickets`, etc.

How to run locally

1. Clone this repo:

git clone https://github.com/<your-username>/CricStars-CWC2023.git
cd CricStars-CWC2023

2. Create a virtual environment (optional but recommended):
  python -m venv venv
  venv\Scripts\activate # on Windows

  source venv/bin/activate # on Linux / Mac

3. Install the requirements:
  pip install streamlit pandas numpy matplotlib seaborn

4. Put your dataset file in the project folder with name `odi_wc_2023.csv`.

5. Run the Streamlit app:
  streamlit run cricstars_app.py

6. Open the link shown in the terminal (usually http://localhost:8501).

## Pages in the app

- **Overview**  
Shows some basic information about the dataset and a simple runs distribution chart.

- **Batting**  
Shows top run scorers, team wise batting stats and a scatter plot of strike rate vs runs.

- **Bowling**  
Shows top wicket takers, team wise bowling stats and a scatter plot of economy vs wickets.

- **Fielding**  
Shows top fielders by total dismissals (catches + stumpings) if those columns exist in your csv.

- **Player Explorer**  
Lets you pick a player and see his batting and bowling summary and small charts.

## Notes

- This is a beginner project, so the code is not perfect.
- I mainly focused on learning the basics of data analysis and dashboards.
- You can fork this repo and improve it by adding more filters, better visuals or match wise analysis.

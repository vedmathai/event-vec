import plotly.express as px
import pandas as pd

df = pd.DataFrame([
    dict(Task="Expected approval<br> from CUREC for the UI", Start='2025-09-01', Finish='2025-09-30', Project="Embedded"),
    dict(Task="Prepare the stimuli<br> for the data collection", Start='2025-09-01', Finish='2025-09-30', Project="Embedded"),
    dict(Task="Perform the data<br> collection on Prolific", Start='2025-10-01', Finish='2025-10-31', Project="Embedded"),
    dict(Task="Perform analysis on<br> the collected<br> update the papers with findings", Start='2025-10-15', Finish='2025-11-15', Project="Embedded"),
    dict(Task="Plan and perform any<br> experiments<br> to further support findings", Start='2025-10-01', Finish='2025-12-31', Project="Other experiments"),
    dict(Task="Update the thesis with<br> any suggestions from the examiners ", Start='2025-11-01', Finish='2025-12-31', Project="Thesis writing"),
    dict(Task="Prepare for submission", Start='2026-01-01', Finish='2026-04-01', Project="Thesis writing"),
])

fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", color="Project", title="Gantt Chart of Tasks")
fig.update_yaxes(autorange="reversed") # otherwise tasks are listed from the bottom up
fig.update_layout(
    font=dict(
        size=24,
    )
)
fig.show()
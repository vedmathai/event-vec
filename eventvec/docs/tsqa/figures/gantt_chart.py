import plotly.express as px
import pandas as pd

df = pd.DataFrame([
    dict(Task="Coding up all relationships <br> for NarrativeGraph", Start='2023-11-01', Finish='2023-11-30', Project="NarrativeGraph"),
    dict(Task="Running the experiments <br> for NarrativeGraph", Start='2023-12-01', Finish='2023-12-31', Project="NarrativeGraph"),
    dict(Task="Coding up the experiments <br> for the Modals Project", Start='2023-11-15', Finish='2023-12-01', Project="Modals"),
    dict(Task="Analyse the experiments for <br> the Modals Project", Start='2023-12-01', Finish='2023-12-31', Project="Modals"),
    dict(Task="Analyse the requirements for <br> data annotation of embedded events", Start='2023-11-01', Finish='2023-11-15', Project="Embedded"),
    dict(Task="Apply for CUREC for the <br> embedded events data", Start='2023-11-15', Finish='2023-12-01', Project="Embedded"),
    dict(Task="Perform data collection", Start='2024-01-01', Finish='2024-01-30', Project="Embedded"),
    dict(Task="Repeat embedded experiments <br> on the collected dataset", Start='2024-02-01', Finish='2024-02-29', Project="Embedded"),

])

fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", color="Project", title="Gantt Chart of Tasks")
fig.update_yaxes(autorange="reversed") # otherwise tasks are listed from the bottom up
fig.update_layout(
    font=dict(
        size=24,
    )
)
fig.show()
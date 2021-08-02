-- displays the max temperature of each state (ordered by State name).
SELECT state, MAX(value) As max_temp FROM temperatures GROUP By state ORDER by state;

import streamlit as st

st.markdown("## Contractor day-rate calculator")
st.text("""This calculator is based on a blog post by Jonathan Sedar.
https://sedar.co/posts/on-contractor-day-rates/""")

currency = st.radio('currency', ('$', '£', '€'))
salary = st.slider('Salary', min_value=35000,
                   max_value=300000, value=90000, step=2500)
st.write("Desired yearly salary:", currency, salary)
days_off = st.slider('days off, including weekends, sick days and holidays',
                     min_value=1, max_value=365, value=140, step=1)
work_days = 365 - days_off
st.write("Work days in a year:", work_days)
utilization = st.slider('predicted utilization',
                        min_value=1, max_value=100, value=80, step=1)
st.write("Utilization: %", (utilization))
corp_tax_rate = st.slider('corp tax rate', min_value=1,
                          max_value=100, value=20, step=1)
st.write("Corp tax rate: %", corp_tax_rate)
st.write("Your desired daily rate is", currency, (salary /
                                                  work_days / (1 - corp_tax_rate / 100) / (utilization / 100)))

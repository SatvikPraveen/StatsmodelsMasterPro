# 🎈 **Streamlit Cheatsheet — Review & Reference**

---

## 🚀 1. **Startup Commands**

| Action              | Command                                                      |
| ------------------- | ------------------------------------------------------------ |
| Run app             | `streamlit run app.py`                                       |
| Set custom port     | `streamlit run app.py --server.port 8501`                    |
| Disable token login | Add `--server.headless true` and `--server.enableCORS false` |
| Clear cache         | `streamlit cache clear`                                      |
| Stop Streamlit      | `Ctrl+C` in terminal                                         |

---

## 🧱 2. **Basic Building Blocks**

```python
import streamlit as st

st.title("My App")
st.header("Header")
st.subheader("Subheader")
st.text("Simple Text")
st.markdown("**Markdown** _text_")
st.code("print('Hello')", language="python")
```

---

## 📥 3. **Widgets & Input**

```python
name = st.text_input("Enter your name")
age = st.number_input("Enter age", min_value=1, max_value=100)
is_checked = st.checkbox("I agree")
option = st.selectbox("Choose", ["A", "B", "C"])
choice = st.radio("Pick one", ["Yes", "No"])
date = st.date_input("Pick a date")
file = st.file_uploader("Upload a CSV", type="csv")
```

---

## 📤 4. **Buttons & Interactions**

```python
if st.button("Click Me"):
    st.success("You clicked the button!")

with st.form("my_form"):
    name = st.text_input("Name")
    submit = st.form_submit_button("Submit")
    if submit:
        st.write("Submitted!")
```

---

## 🖼️ 5. **Display Data**

```python
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(5, 2), columns=['A', 'B'])
st.dataframe(df)
st.table(df.head())
st.json({'key': 'value'})
```

---

## 📊 6. **Charts & Visuals**

```python
st.line_chart(df)
st.bar_chart(df)
st.area_chart(df)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(df['A'], df['B'])
st.pyplot(fig)

import plotly.express as px
fig = px.scatter(df, x='A', y='B')
st.plotly_chart(fig)
```

---

## 📌 7. **Layouts & Containers**

```python
st.sidebar.title("Sidebar")
st.sidebar.slider("Slider", 0, 100)

col1, col2 = st.columns(2)
with col1:
    st.write("Left")
with col2:
    st.write("Right")

with st.expander("More options"):
    st.write("Hidden text here")

with st.container():
    st.write("Grouped content")
```

---

## 🧠 8. **State Management (Session State)**

```python
if 'count' not in st.session_state:
    st.session_state.count = 0

if st.button("Increment"):
    st.session_state.count += 1

st.write("Count:", st.session_state.count)
```

---

## 💾 9. **Cache & Performance**

```python
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

@st.cache_resource
def get_model():
    return load_heavy_model()
```

---

## 🛠️ 10. **Configuration (optional)**

Create a file called `.streamlit/config.toml`:

```toml
[server]
headless = true
port = 8501
enableCORS = false
```

---

## 🐳 11. **Streamlit in Docker**

```Dockerfile
EXPOSE 8501
CMD ["streamlit", "run", "Home.py", "--server.headless=true", "--server.port=8501"]
```

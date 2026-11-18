import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title='Logistic Map AI Lab', page_icon='🌀', layout='wide')
st.markdown('''<style>
.block-container {padding-top: 1.5rem; max-width: 1250px;}
[data-testid="stMetric"] {background:rgba(110,130,180,.10);padding:14px;border-radius:12px;}
</style>''', unsafe_allow_html=True)
st.title('🌀 Logistic Map AI Lab 2.0')
st.caption('Interactive neural-network predictions of periodic and chaotic dynamics')
st.latex(r'x_{n+1}=r x_n(1-x_n)')

@st.cache_resource(show_spinner='Training the MLP model (first run only)...')
def train_model():
    rng = np.random.default_rng(42)
    # Diverse one-step transitions; the MLP learns (r, x_n) -> x_(n+1).
    r = rng.uniform(2.5, 4.0, 12000)
    x = rng.uniform(0.001, 0.999, 12000)
    X = np.column_stack((r, x))
    y = r * x * (1 - x)
    model = make_pipeline(StandardScaler(), MLPRegressor(
        hidden_layer_sizes=(64, 64), activation='relu',
        max_iter=700, early_stopping=True, n_iter_no_change=20,
        random_state=42))
    model.fit(X, y)
    return model

model = train_model()

with st.sidebar:
    st.header('⚙️ Simulation settings')
    preset = st.selectbox('Regime', ['Custom', 'Periodic (r = 3.2)', 'Period-4 (r = 3.5)', 'Chaotic (r = 3.9)'])
    preset_r = {'Custom':3.5, 'Periodic (r = 3.2)':3.2, 'Period-4 (r = 3.5)':3.5, 'Chaotic (r = 3.9)':3.9}[preset]
    r = st.slider('Growth parameter r', 2.5, 4.0, preset_r, 0.01)
    x0 = st.number_input('Initial condition x₀', 0.001, 0.999, 0.23, 0.01, format='%.3f')
    steps = st.slider('Number of iterations', 10, 200, 50, 5)
    mode = st.radio('Prediction mode', ['One-step (true previous state)', 'Recursive (predicted previous state)'])
    st.caption('One-step measures local accuracy; recursive forecasts reveal error accumulation.')
    st.button('🔮 Generate prediction', type='primary', use_container_width=True)

actual = np.empty(steps + 1)
actual[0] = x0
for i in range(steps):
    actual[i+1] = r * actual[i] * (1 - actual[i])

pred = np.empty(steps + 1)
pred[0] = x0
for i in range(steps):
    previous = actual[i] if mode.startswith('One-step') else pred[i]
    pred[i+1] = float(model.predict(np.array([[r, previous]]))[0])
    if not mode.startswith('One-step'):
        pred[i+1] = np.clip(pred[i+1], 0.0, 1.0)

errors = np.abs(actual[1:] - pred[1:])
mae = mean_absolute_error(actual[1:], pred[1:])
rmse = np.sqrt(mean_squared_error(actual[1:], pred[1:]))
regime_label = 'Periodic / transition' if r < 3.57 else 'Potentially chaotic'
a, b, c, d = st.columns(4)
a.metric('MAE', f'{mae:.5f}')
b.metric('RMSE', f'{rmse:.5f}')
c.metric('Iterations', steps)
d.metric('Regime (approx.)', regime_label)
st.info('For chaotic dynamics, small errors can grow rapidly in recursive forecasts. The regime label is approximate, not a rigorous classification.')

plt.style.use('default')
def newfig():
    fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
    return fig, ax

left, right = st.columns(2)
with left:
    st.subheader('📈 Real vs. predicted trajectory')
    fig, ax = newfig()
    ax.plot(actual, label='Real', linewidth=2)
    ax.plot(pred, '--', label='MLP prediction', linewidth=1.6)
    ax.set(xlabel='Iteration n', ylabel='State x', ylim=(-0.08, 1.08))
    ax.legend()
    ax.grid(alpha=.25)
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.subheader('📉 Absolute prediction error')
    fig, ax = newfig()
    ax.plot(np.arange(1, steps+1), errors, color='#dc6b35')
    ax.set(xlabel='Iteration n', ylabel='Absolute error')
    ax.grid(alpha=.25)
    st.pyplot(fig)
    plt.close(fig)

left2, right2 = st.columns(2)
with left2:
    st.subheader('🔁 Return map')
    fig, ax = newfig()
    xx = np.linspace(0, 1, 300)
    ax.plot(xx, r * xx * (1-xx), label='Logistic function')
    ax.scatter(actual[:-1], actual[1:], s=14, alpha=.6, label='Trajectory')
    ax.plot(xx, xx, ':', color='gray', label='y = x')
    ax.set(xlabel='xₙ', ylabel='xₙ₊₁')
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)
    st.pyplot(fig)
    plt.close(fig)
with right2:
    st.subheader('🌌 Bifurcation diagram')
    @st.cache_data(show_spinner=False)
    def bifurcation():
        rs = np.linspace(2.5, 4.0, 600)
        xs = np.full_like(rs, .23)
        for _ in range(350):
            xs = rs * xs * (1-xs)
        r_out, x_out = [], []
        for _ in range(70):
            xs = rs * xs * (1-xs)
            r_out.append(rs.copy())
            x_out.append(xs.copy())
        return np.concatenate(r_out), np.concatenate(x_out)
    rs, xs = bifurcation()
    fig, ax = newfig()
    ax.plot(rs, xs, ',k', alpha=.3)
    ax.axvline(r, color='#dc6b35', linewidth=1.8, label=f'r = {r:.2f}')
    ax.set(xlabel='Parameter r', ylabel='Long-run x')
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

st.subheader('📋 Prediction data')
df = pd.DataFrame({'iteration': np.arange(steps+1), 'real': actual,
                   'mlp_prediction': pred, 'absolute_error': np.r_[0.0, errors]})
st.dataframe(df, use_container_width=True, hide_index=True)
st.download_button('⬇️ Download results (CSV)', df.to_csv(index=False).encode('utf-8'),
                   file_name='logistic_predictions.csv', mime='text/csv')
with st.expander('ℹ️ About the model and limitations'):
    st.markdown('''**Model:** MLPRegressor with two hidden layers (64, 64), trained on 12,000 synthetic one-step logistic-map samples for r in [2.5, 4.0]. Input features are (r, xₙ); target is xₙ₊₁. This is a new demonstration model, not necessarily the model from the earlier research project.\n\n**One-step:** each prediction receives the true preceding state. **Recursive:** each prediction receives the previous predicted state. Recursive errors may accumulate strongly in chaotic regimes.\n\n**Evaluation:** MAE and RMSE shown here are errors on the selected simulated trajectory, not independent held-out test-set metrics.''')

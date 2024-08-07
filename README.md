# Options Pricer
[Web app](https://www.emanuelmalek.com/quant_projects/options_pricer.html) for computing the price of options and the PnL as volatility and spot change.

## Functionalities
- Compute Call/Put prices using Black-Scholes and Binomial Tree Methods.
- Calculate PnL of the Call/Put as spot and volatility change and display in a heatmap.
- Can handle European & American Vanilla, Digital and Barrier options.
- Black-Scholes calculation can be run in vectorised form on an array of options parameters.
- Binomial Tree calculation is performed in vectorised form for each value of option parameters.

## How to run
The web app is built using streamlit. After pip installing streamlit, you can launch the web app by running
```
streamlit run main.py
```

## Future plans
- Include other pricing methods, e.g. Monte Carlo.
- Include other Exotic options.

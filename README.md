# Sona Power Predict - 2026

**College Name:** Sona College of Technology  
**Team Name:** Flyon 

### Team Members
* **Pragadeesh V** - Year 2, Computer Science and Engineering(Artificial Intelligence and Machine Learning)
* **Nithish J** - Year 2, Computer Science and Engineering(Artificial Intelligence and Machine Learning)
* **Monish R** - Year 2, Computer Science and Engineering(Artificial Intelligence and Machine Learning)
* **Rithik R** - Year 2, Computer Science and Engineering(Artificial Intelligence and Machine Learning)

---

### Libraries Used in Model
Based on the `mymodelfile.py` submission, the following Python libraries are utilized for data manipulation and mathematical operations:
* **`pandas`**:Loading CSV files, groupby aggregations, time-series operations, building the innings-level dataset
* **`numpy`**: Array operations, EMA calculations, clipping predictions, error metrics
* **`scikit`**: Three things: HuberRegressor (robust linear model), HistGradientBoostingRegressor (tree ensemble), StandardScaler (feature normalisation)
* **`scipy`**: difflib.get_close_matches is from stdlib — scipy is imported but used via minimize for weight optimisation in earlier versions
* **`difflib`**: Fuzzy player name matching — resolves "Vaibhav Suryavanshi" to "V Suryavanshi" in training data
---

### Model
* **HuberRegressor**
Tree models were tested exhaustively and all failed with MAE of 12–14 and bias of −8 to −10 runs. The reason: IPL powerplay scores have been rising every year (2008: 46 avg → 2025: 57 avg → 2026: 61 avg). Tree models cannot extrapolate beyond their training range — they always predict the historical mean.
HuberRegressor on residuals solves this. Since the era anchor absorbs the current scoring level, the residual is always centred near zero regardless of the year. The linear model then predicts deviations from that anchor, which it can do correctly.

### License
This project is licensed under the **MIT License**.

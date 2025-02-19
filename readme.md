
# Team Ordinals - Real-Time Bidding (RTB) Pipeline

Welcome to the repository for Team Ordinals' RTB pipeline. This project addresses the challenge of predicting Click-Through Rate (CTR) and Conversion Rate (CVR) for real-time bidding scenarios and then dynamically adjusting bids using a Reinforcement Learning (RL) based strategy.

---

## 1. Team Details

**Team Name:** Team Ordinals  
**Team Members:**  
- Tarun Jain  
- Ujjwal Kakar  
- Aryan Sood  
- Utkarsh Chaudhary  

---

## 2. Code Instructions

To implement the bidder interface, we first recommend initializing a virtual environment to avoid having any dependency conflicts with your current installation of Python.

1. **Create a virtual environment (optional)**
   ```bash
   python -m venv ./venv
   ```
2. **Now activate the venv**  
   - **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - **Linux / macOS**:
     ```bash
     source venv/bin/activate
     ```
3. **Install all the requirements** listed in `requirements.txt`
   ```bash
   python -m pip install -r requirements.txt
   ```
4. **Import** the `OrdinalBidder` from the `bidder` package
   ```python
   from bidder.ordinal import OrdinalBidder
   from bidder.BidRequest import BidRequest

   bidding_agent = OrdinalBidder()
   ```
5. **Use** the `bidding_agent` for prediction by calling its function:
   ```python
   ordinal_bidder_bid = bidding_agent.getBidPrice(bidRequest)
   ```

---

## 3. Approach Description

Our solution addresses a Real-Time Bidding (RTB) problem where we predict both CTR and CVR, and then use a Reinforcement Learning (RL) based bidding strategy to adjust bids dynamically. The pipeline is designed to handle varying constraints such as budget and time, ensuring optimal ad placements and efficient budget utilization.

### 3.1 Data Engineering

- **Data Cleaning & Imputation:**  
  Remove anomalies, handle missing values, and impute relevant fields to ensure consistent inputs.

- **Target Encoding & Scaling:**  
  High-cardinality categorical features (e.g., `creative_id`, `site_id`) are target-encoded. Numerical features are scaled as needed to aid model convergence.

- **Creation of New Features:**  
  - **Aspect Ratio:** Consolidates ad slot dimensions into a single feature, reducing dimensionality while preserving essential ad display properties.  
  - **Creative-User Affinity:** Captures how often a user interacts with a specific creative, drawing on collaborative filtering concepts.  
  - **Time/Contextual Features:** Includes weekday, hour-of-day, and seasonal indicators to capture user behavior context.

- **Polars for Data Manipulation (Latency Optimization):**  
  We leverage the **Polars** library for faster data processing (~2x faster than Pandas), which is crucial given the large-scale RTB data.

---

## 4. Exploratory Data Analysis (EDA)

1. **Data Size & Imbalance:**
   - **Bid_Data:** ~53 million records  
   - **Impression_Data:** ~12 million records (only ~22.9% of bids lead to an impression)  
   - **Click_Data:** ~10 thousand records  
   - **Conversion_Data:** ~494 records  

   The dataset is highly sparse with extremely imbalanced targets (CTR ≈ 0.08%, CVR on clicks ≈ 0.05%).

2. **User Behavior & Contextual Insights:**  
   - Clear relationships between CTR and features such as **weekday**, **time of day**, and **advertiser ID**.  
   - Some advertisers exhibit higher conversion rates with fewer ads, indicating strong user affinity.

3. **Creative & Adslot Correlation:**  
   - Heatmaps revealed strong correlations between certain `creative_id` values and specific adslots.  
   - The **aspect ratio** of ad slots provides a concise yet effective representation of ad layout.

Given the imbalanced nature of the data, precision is often more critical than recall to minimize wasted impressions. We emphasize metrics like the F1-score and weighted harmonic means to evaluate performance.

---

## 5. Feature Engineering

1. **Aspect Ratio**  
   - Converts raw adslot dimensions into a single ratio-based feature, reducing dimensionality while capturing the essence of the display.

2. **Creative-User Affinity**  
   - Measures the frequency of user interactions with specific creatives, leveraging collaborative filtering principles.

3. **Time-Based Features**  
   - Incorporates indicators such as the hour of the day and weekday/weekend flags, along with seasonal patterns if applicable.

4. **Contextual Features**  
   - Uses advertiser IDs, historical CTR/CVR data, and site category or publisher contexts (when available).

These engineered features, along with the cleaned and scaled original data, feed into our ML pipeline for CTR and CVR prediction.

---

## 6. Model Selection

### 6.1 Ensemble for CTR & CVR

We use an ensemble approach combining two primary models:

1. **Gradient Boosting Machines (GBMs):**  
   - Capable of capturing non-linear relationships and handling sparse data effectively.
   
2. **Logistic Regression:**  
   - Offers a fast and interpretable baseline, complementing GBMs to enhance robustness.
  
     
We use LightGBM Regressor as the bidding model due to its efficiency in handling large datasets and its ability to capture complex patterns in budget allocation and bid aggression. LightGBM’s gradient boosting approach allows for fast training and prediction, making it ideal for real-time bidding strategies.

By integrating GBMs and Logistic Regression, we leverage the unique strengths of each algorithm to capture diverse predictive signals, resulting in improved overall accuracy for CTR and CVR predictions.

### 6.2 Final Bidding Strategy (RL-Based)

1. **FAB (Reinforcement Learning) Model:**  
   - Dynamically adjusts bid prices based on real-time auction conditions, balancing budget constraints with impression opportunities.

2. **Modified FAB:**  
   - **Policy Replacement:** Instead of using complex neural models, we employ an ensemble of Linear and Polynomial Regression to approximate Q-values or direct bidding signals.  
   - **Reduced Latency:** This approach avoids heavy matrix operations, ensuring real-time decision-making within <3ms.

3. **Adaptive Bid Flow:**  
   - Modulates bidding aggression based on remaining budget and campaign time:
     - **Passive Bidding:** When budget is high and time is ample, bids are conservative to gather more data.
     - **Aggressive Bidding:** When budget is high but time is short, bids are increased to secure more impressions quickly.
   - This strategy dynamically balances cost efficiency with exposure.

**Rationale for Simpler Models:**  
- Complex neural networks can introduce latency challenges in real-time bidding.  
- Simpler models, with optimized data processing and streamlined RL components, provide rapid adaptation with minimal computational overhead.

---

## 7. Hyperparameter Tuning

- **GBMs (LightGBM / XGBoost):**  
  - Use Grid search or Bayesian optimization to tune learning rate, max_depth, and n_estimators, with early stopping based on validation AUC.

- **Logistic Regression:**  
  - Tune the regularization parameter (C) to manage the bias-variance trade-off.

- **Reinforcement Learning Parameters (FAB / Modified FAB):**  
  - Optimize learning rate, exploration-exploitation balance, and reward shaping to ensure near real-time performance.

---

## 8. Evaluation Strategy

1. **Offline Model Evaluation:**  
   - Metrics such as **ROC-AUC** and **RMSE** are used for CTR and CVR predictions.
   - Confusion matrix derivatives (precision, recall, F1-score) are employed to evaluate performance on imbalanced classes.

2. **Online / Simulated Environment:**  
   - Evaluate budget utilization against conversion rate or effective CPM.
   - Measure Return on Investment (ROI) for different bidding strategies.
   - Monitor latency to ensure predictions meet real-time thresholds.

3. **Validation Results (Sample)**  

   | Adv.* (ID) | RMSE (CTR/CVR) | ROC-AUC |
   |------------|---------------:|--------:|
   | 1458       |           0.015 |     0.92|
   | 3476       |           0.014 |     0.88|
   | 3427       |           0.016 |     0.93|
   | 3386       |           0.024 |     0.84|
   | 3358       |           0.083 |     0.82|
   | **Total**  |       **0.021** | **0.985**|

   > *Adv.: Advertiser ID

These results highlight the improved performance achieved by our ensemble approach for CTR/CVR prediction and the effectiveness of the RL-based bidding strategy.

---

## 9. Additional Relevant Information

- **Handling Sparse Data:**  
  Sampling strategies and weighting methods are implemented to manage the imbalanced distribution of clicks and conversions.

- **Latency Considerations:**  
  The design ensures sub-100ms decision-making, critical for real-time bidding. This is achieved through optimized data processing with Polars and the use of simplified RL models.

- **Scalability:**  
  The pipeline is built to efficiently handle tens of millions of bids. Engineered features are designed to be lightweight yet informative.

- **Edge Case Management:**  
  - **Budget Constraints:** When the budget nears depletion, the strategy becomes more conservative.  
  - **Time Constraints:** As the campaign end approaches, the strategy shifts to more aggressive bidding to meet impression or click targets.

---

## 10. Conclusion

Team Ordinals presents a hybrid pipeline that combines:

1. **Ensemble Modeling** (using GBMs and Logistic Regression) for accurate CTR and CVR predictions.  
2. A **Reinforcement Learning (Modified FAB) strategy** for dynamic, real-time bid adjustments that optimize both ROI and resource utilization.  
3. An **Adaptive Bid Flow** mechanism to effectively handle varying budget and time constraints.

This comprehensive approach balances prediction accuracy, speed, and practical constraints in the competitive RTB environment.

---

**Thank you for exploring our repository!**  
For questions or further information, please feel free to reach out to **Team Ordinals**.
```

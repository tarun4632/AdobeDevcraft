# Team Details
## **Team Name**
<div style="font-size:25px;">
    Team Ordinals
</div>

## **Team Members**
1. Ujjwal Kakar
2. Aryan Sood
3. Tarun Jain
4. Utkarsh Chaudhary

# Code Instructions

To implement the bidder interface, we first recommend initializing a virtual environment to avoid having any dependency conflicts with your current installation of python.
1. Create a virtual environment (optional)
<pre>
python -m venv ./venv
</pre>
2. Now activate the venv
<pre>
venv\Scripts\activate             # for windows
source venv/bin/activate          # for linux and mac os
</pre>
3. Install all the requirements of the project mentioned in ```requirements.txt```
<pre>
python -n pip install -r requirements.txt
</pre>
4. Import the ```OrdinalBidder``` from the ```bidder``` package
<pre>
from bidder.ordinal import OrdinalBidder
from bidder.BidRequest import BidRequest

bidding_agent = OrdinalBidder()
</pre>

4. Use the ```bidding_agent``` for prediction by calling function.
<pre>
ordinal_bidder_bid = bidding_agent.getBidPrice(bidRequest)
</pre>

# Approach Description




# Exploratory Data Analysis (EDA)

Observation: bid dataframe is the largest, and not all bids lead to an impression as the impression database is roughly 22.9% and CTR on impressions is around 0.08%. Conversion Rate on click is 0.05% and on impression is 0.004% 

Sizes:
1. Bid_Data: 53_289_330
2. Impression_Data: 12_237_087
3. Click_Data: 9_978
4. Conversion_Data: 494

In all, we can say the data is sparse, and highly imbalanced target classes, making it essential we train on a Confusion Matrix derivative.

Since this is a bidding game, we realise the importance of precision is more than recall and we will use a weighted harmonic sum of the two as our maximisation parameter.

## 


# Feature Selection and Engineering
Alot of features which are hashed or anonymised will become redundant in any form of model training.

We perform transformation
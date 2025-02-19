#!/usr/bin/python
# -*- coding:utf8 -*-
import random
import numpy as np
import pickle
import json
import logging
import datetime
from sklearn.preprocessing import MinMaxScaler
from category_encoders import TargetEncoder
import pandas as pd
import os
import time
from bidRequest import BidRequest
os.environ["LOKY_MAX_CPU_COUNT"] = "6"

class Bid(object):

    def __init__(self, ctr_threshold=0.5, cvr_threshold=0.5):
        # Load encoders and scalers
        with open('saved/encoders_scalers.pkl', 'rb') as f:
            enscdict = pickle.load(f)
            self.te = enscdict['te']
            self.mms = enscdict['mms']
        
        # Load necessary mappings
        with open('saved/creative_userid_count.json') as f:
            self.creative_userid_dict = json.load(f)
        with open('saved/creative_adtype_map.json', 'r') as f:
            self.advertiser_bias = json.load(f)
        
        self.ctr_threshold = ctr_threshold
        self.cvr_threshold = cvr_threshold
        
        # Note: Budget is now managed externally in the evaluation code.
        
        # Load all CTR models
        self.ctrmodels = {}
        model_ids = ['1458', '3358', '3386', '3427', '3476']
        for model_id in model_ids:
            with open(f'model/{model_id}.pkl', 'rb') as f:
                self.ctrmodels[model_id] = pickle.load(f)
        
        # Load CVR models
        self.cvrmodels = {}
        cvr_model_ids = ['3358', '3476']
        for model_id in cvr_model_ids:
            with open(f'model/{model_id}_cvr.pkl', 'rb') as f:
                self.cvrmodels[model_id] = pickle.load(f)
        
        # Load bid model
        with open('model/bid.pkl', 'rb') as f:
            self.bidmodel = pickle.load(f)

    def get_bid_price(self, bidRequest: BidRequest) -> int:
        try:
            # Get floor price from bid request
            floor_price = int(bidRequest.ad_slot_floor_price)
            
            # Extract features
            creative_entry = self.creative_userid_dict.get(bidRequest.creative_id, {})
            user_tags = bidRequest.user_tags.split(',') if bidRequest.user_tags else []
            
            feature_dict = {
                'region': [bidRequest.region],
                'city': [bidRequest.city],
                'exchange': [int(bidRequest.ad_exchange)],
                'width': [int(bidRequest.ad_slot_width) / 100],
                'height': [int(bidRequest.ad_slot_height) / 100],
                'visibility': [bidRequest.ad_slot_visibility],
                'format': [bidRequest.ad_slot_format],
                'floor_price': [floor_price],
                'device': [bidRequest.device],
                'browser': [bidRequest.browser],
                'ad_type': [bidRequest.ad_type],
                'day_of_week': [datetime.datetime.strptime(bidRequest.timestamp[:8], "%Y%m%d").weekday()],
                'creative_user_count': [sum(creative_entry.get(tag, 0) for tag in user_tags)],
                'time_block': [int(bidRequest.timestamp[8:10]) * 4 + int(bidRequest.timestamp[10:12]) // 15]
            }
            
            feature_df = pd.DataFrame(feature_dict)
            # Apply target encoding and scaling
            encoded_features = self.te.transform(feature_df)
            scaled_features = self.mms.transform(encoded_features)
            
            advertiser_id = bidRequest.advertiser_id
            if advertiser_id not in self.ctrmodels:
                logging.info("No CTR model available for advertiser %s", advertiser_id)
                return -1  # No model available for this advertiser
            
            # Predict CTR and check threshold
            ctr = self.ctrmodels[advertiser_id].predict(scaled_features)[0]
            if ctr < self.ctr_threshold:
                logging.info("CTR below threshold: %s < %s", ctr, self.ctr_threshold)
                return -1
            
            # Predict CVR (if available) and check threshold
            cvr = None
            if advertiser_id in self.cvrmodels:
                cvr = self.cvrmodels[advertiser_id].predict(scaled_features)[0]
                if cvr < self.cvr_threshold:
                    logging.info("CVR below threshold: %s < %s", cvr, self.cvr_threshold)
                    return -1
            
            # Get base predicted bid
            predicted_bid = self.bidmodel.predict(scaled_features)[0]
            
            # --- Dynamic Bid Adjustment ---
            # Compute an adjustment factor based on how much the predicted CTR (and CVR) exceed the thresholds.
            bid_adjustment = 1.0 + 0.5 * (ctr - self.ctr_threshold)
            if cvr is not None:
                bid_adjustment += 0.5 * (cvr - self.cvr_threshold)
            # Limit the adjustment to a reasonable range (e.g., between 0.8 and 1.2)
            bid_adjustment = max(0.8, min(bid_adjustment, 1.2))
            adjusted_bid = predicted_bid * bid_adjustment
            
            # Final bid is at least 1.1 * floor_price or the adjusted bid, whichever is higher.
            final_bid = int(max(1.1 * floor_price, adjusted_bid))
            
            # Log the bid price; external evaluation code will handle budgets.
            logging.info("Bid placed: %s", final_bid)
            
            return final_bid
        
        except Exception as e:
            logging.error(f'Error in get_bid_price: {e}')
            return 100  # Default bid in case of error

logging.basicConfig(filename='bidder.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

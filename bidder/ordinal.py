from BidRequest import BidRequest
from Bidder import Bidder
import random
import numpy as np
import pickle
import json
import logging
import datetime

logging.basicConfig(filename='bidder.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class OrdinalBidder(Bidder):
    def __init__(self,ctr_threshold=0.7,cvr_threshold=0.7):
        with open('saved/target_encoders.pkl','rb') as f:
            te = pickle.load(f) 
            self.target_region = te['region']
            self.target_city = te['city']
            self.target_device = te['device']
            self.target_browser = te['browser']
            self.target_adType = te['ad_type']
            self.week_bias = te['week']
        with open('saved/creative_userid_count.json') as f: 
            self.creative_userid_dict = json.load(f)
        with open('saved/creative_adtpye_map','r') as f: 
            self.advertiser_bias = json.load(f)
        self.ctr_threshold = ctr_threshold
        self.cvr_threshold = cvr_threshold
        #TODO
        self.ctrmodels = {
            '1458': None,
            '3358': None,
            '3386': None,
            '3427': None,
            '3476': None,
        }
        self.cvrmodels = {
            '3358':None,
            '3476':None
        }
        self.bidmodel = None

    def transform_input_to_useful(self,bidRequest : BidRequest) -> np.array :
        bidid = bidRequest.bidId
        region = bidRequest.region
        city = bidRequest.city
        exchange = bidRequest.adExchange
        width = bidRequest.adSlotWidth
        height = bidRequest.adSlotHeight
        visibility = bidRequest.adSlotVisibility
        format = bidRequest.adSlotFormat
        floor_price = bidRequest.adSlotFloorPrice
        creative_user_count = sum([self.creative_userid_dict[bidRequest.creativeID][k] for k in bidRequest.userTags.split(",")])
        advertiser_id = bidRequest.advertiserId
        time_block = int(bidRequest.timestamp[8:10])*4+int(bidRequest.timestamp[10:12])//15
        day = datetime.datetime.strptime(bidRequest.timestamp[:7]).day
        ua = bidRequest.userAgent.lower()
        device = 'ios' if 'ios' in ua or 'iphone' in ua or 'ipad' in ua or 'ipod' in ua else 'mac' if 'macintosh' in ua or 'mac' in ua or 'darwin' in ua else \
            'windows' if 'windows' in ua else 'linux' if 'linux' in ua else 'android' if 'android' in ua else 'other'
        browser = "safari" if "safari" in ua else "chrome" if "chrome" in ua else 'firefox' if 'firefox' in ua or 'mozilla' in ua else 'edge' if \
            'edge' in ua else 'ie' if 'msie' in ua or 'trident' in ua else 'other'
        
        adtype = "banner" if width / height > 3 else "rectangle" if 1.2 < width / height < 2 else "square" if 0.8 <= width / height <= 1.2 else "vertical" if height / width > 1.2 else "other"
    
        return [bidid , region , city , exchange , width , height , visibility , format ,floor_price , creative_user_count , advertiser_id , time_block,
                day,device ,browser , adtype]
        
        
    def getBidPrice(self, bidRequest : BidRequest) -> int:
        try:
            details = self.transform_input_to_useful(bidRequest)
        except Exception as e:
            logging.error(f'Error while transforming input: {e}')
            return -1
        
        """details = {
            0 : bidId
            1 : region
            2 : city
            3 : exchange
            4 : width
            5 : height
            6 : visibility
            7 : format
            8 : floor_price
            9 : creative_user_count
            10: advertiser_id
            11: time_block
            12: day_of_week
            13: device
            14: browser
            15: ad_type
        }"""
        ctr = self.ctrmodel[details[10]](np.array(details[1:8]+details[9:10]+details[11:16]))
        if ctr < self.ctr_threshold: return -1
        if details[10] in self.cvrmodels: 
            cvr = self.cvrmodels[details[10]](np.array(details[1:8]+details[9:10]+details[11:16]))
            return self.bidmodel() #bidmodel with cvr get 
        else:
            return self.bidmodel() #bidmodel with cvr 0
        
        
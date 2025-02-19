# from BidRequest import BidRequest
# from Bidder import Bidder
# import random
# import numpy as np
# import pickle
# import json
# import logging
# import datetime
# from sklearn.preprocessing import MinMaxScaler
# from category_encoders import TargetEncoder
# import pandas as pd
# import os
# import time
# os.environ["LOKY_MAX_CPU_COUNT"] = "6"  # Replace 4 with the number of physical cores

# logging.basicConfig(filename='bidder.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# class OrdinalBidder(Bidder):
#     def __init__(self,ctr_threshold=0.7,cvr_threshold=0.7):
#         with open('saved/encoders_scalers.pkl','rb') as f:
#             enscdict = pickle.load(f) 
#             self.te = enscdict['te']
#             self.mms = enscdict['mms']
#         with open('saved/creative_userid_count.json') as f: 
#             self.creative_userid_dict = json.load(f)
#         with open('saved/creative_adtype_map.json','r') as f: 
#             self.advertiser_bias = json.load(f)
#         self.ctr_threshold = ctr_threshold
#         self.cvr_threshold = cvr_threshold
        
#         self.ctrmodels = {}
#         with open('model/1458.pkl','rb') as f:
#             self.ctrmodels['1458'] = pickle.load(f)
            
#         with open('model/3358.pkl','rb') as f:
#             self.ctrmodels['3358'] = pickle.load(f)
            
#         with open('model/3386.pkl','rb') as f:
#             self.ctrmodels['3386'] = pickle.load(f)
            
#         with open('model/3427.pkl','rb') as f:
#             self.ctrmodels['3427'] = pickle.load(f)
            
#         with open('model/3476.pkl','rb') as f:
#             self.ctrmodels['3476'] = pickle.load(f)    
            
#         self.cvrmodels = {}
#         with open('model/3476_cvr.pkl','rb') as f:
#             self.ctrmodels['3476'] = pickle.load(f)  
            
#         with open('model/3358_cvr.pkl','rb') as f:
#             self.ctrmodels['3358'] = pickle.load(f)  
            
        
#         with open('model/bid.pkl','rb') as f:
#             self.bidmodel = pickle.load(f)  
            
            
#     def transform_input_to_useful(self,bidRequest : BidRequest) -> np.array :
#         bidid = bidRequest.bidId
#         region = bidRequest.region
#         city = bidRequest.city
#         exchange = int(bidRequest.adExchange)
#         width = int(bidRequest.adSlotWidth)/100
#         height = int(bidRequest.adSlotHeight)/100
#         visibility = bidRequest.adSlotVisibility
#         format = bidRequest.adSlotFormat
#         floor_price = int(bidRequest.adSlotFloorPrice)
#         creative_user_count = sum([self.creative_userid_dict[bidRequest.creativeID].get(k,0) for k in bidRequest.userTags.split(",")])
#         advertiser_id = bidRequest.advertiserId
#         time_block = int(bidRequest.timestamp[8:10])*4+int(bidRequest.timestamp[10:12])//15
#         day = str(datetime.datetime.strptime(bidRequest.timestamp[:8],"%Y%m%d").day)
#         ua = bidRequest.userAgent.lower()
#         device = 'ios' if 'ios' in ua or 'iphone' in ua or 'ipad' in ua or 'ipod' in ua else 'mac' if 'macintosh' in ua or 'mac' in ua or 'darwin' in ua else \
#             'windows' if 'windows' in ua else 'linux' if 'linux' in ua else 'android' if 'android' in ua else 'other'
#         browser = "safari" if "safari" in ua else "chrome" if "chrome" in ua else 'firefox' if 'firefox' in ua or 'mozilla' in ua else 'edge' if \
#             'edge' in ua else 'ie' if 'msie' in ua or 'trident' in ua else 'other'
        
#         adtype = "banner" if width / height > 3 else "rectangle" if 1.2 < width / height < 2 else "square" if 0.8 <= width / height <= 1.2 else "vertical" if height / width > 1.2 else "other"
    
#         encoded_df = self.te.transform(pd.DataFrame({
#             'region': [region], 
#             'city': [city], 
#             'device': [device], 
#             'browser': [browser], 
#             'ad_type': [adtype], 
#             'day_of_week': [day]
#         }))

#         region, city, device, browser, adtype, day = np.array([
#             encoded_df['region'].iloc[0], 
#             encoded_df['city'].iloc[0], 
#             encoded_df['device'].iloc[0], 
#             encoded_df['browser'].iloc[0], 
#             encoded_df['ad_type'].iloc[0], 
#             encoded_df['day_of_week'].iloc[0]
#         ])*100

        
#         # Transform numerical features using Min-Max scaling
#         scaled_df = self.mms.transform(pd.DataFrame({
#             'creative_user_count': [creative_user_count], 
#             'time_block': [time_block]
#         }))

#         # Extract and scale values
#         creative_user_count, time_blocks = scaled_df.flatten() 
    
#         return [region , city , exchange , width , height , visibility , format , creative_user_count , time_block,
#                 day,device ,browser , adtype] , str(advertiser_id) , floor_price
        
        
#     def getBidPrice(self, bidRequest : BidRequest) -> int:
#         # try:
#         X , id , floor = self.transform_input_to_useful(bidRequest)
#         # except Exception as e:
#         #     logging.error(f'Error while transforming input: {e}')
#         #     return 100
        
#         """details = {
#             0 : bidId
#             1 : region
#             2 : city
#             3 : exchange
#             4 : width
#             5 : height
#             6 : visibility
#             7 : format
#             8 : floor_price
#             9 : creative_user_count
#             10: advertiser_id
#             11: time_block
#             12: day_of_week
#             13: device
#             14: browser
#             15: ad_type
#         }"""
#         # print(X)
#         factor = 1
#         ctr = self.ctrmodels[id].predict((np.array([X])))
#         if ctr < self.ctr_threshold: return -1
#         if id in self.cvrmodels: 
#             cvr = self.cvrmodels[id].predict((np.array([X+[int(id)]])))
#             if cvr > self.cvr_threshold: factor = 1.5
#         return max(1.1*floor , self.bidmodel.predict(np.array([X+[int(id)]]))*factor)[0]
    
# if __name__ == '__main__':
#     bidRequest = BidRequest()
#     bidRequest.bidId = "b382c1c156dcbbd5b9317cb50f6a747b"
#     bidRequest.timestamp = "20130606000104008"
#     bidRequest.visitorId = "Vh16OwT6OQNUXbj"
#     bidRequest.userAgent = "mozilla/4.0 (compatible; msie 6.0; windows nt 5.1; sv1; qqdownload 718"
#     bidRequest.ipAddress = "192.168.0.1"
#     bidRequest.region = 80
#     bidRequest.city = 87
#     bidRequest.adExchange = 1
#     bidRequest.domain = "tFKETuqyMo1mjMp45SqfNX"
#     bidRequest.url = "249b2c34247d400ef1cd3c6bfda4f12a"
#     bidRequest.adSlotID = "mm_11402872_1272384_3182279"
#     bidRequest.adSlotWidth = 300
#     bidRequest.adSlotHeight = 250
#     bidRequest.adSlotVisibility = 1
#     bidRequest.adSlotFormat = 1
#     bidRequest.adSlotFloorPrice = 0
#     bidRequest.creativeID = "00fccc64a1ee2809348509b7ac2a97a5"
#     bidRequest.advertiserId = "3427"
#     bidRequest.userTags = ""
    
#     bidder = OrdinalBidder()
#     start = time.time()
#     v = bidder.getBidPrice(bidRequest)
#     end = time.time()
#     print(f'Bid price: {v}, Time taken: {end-start} seconds')
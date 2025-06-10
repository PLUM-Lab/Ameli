from lxml.html import fromstring
from itertools import chain
from tqdm import tqdm
import requests
import logging
import re
import logging 
import time
import re
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
import urllib.parse
import urllib
headers = {'User-Agent': 'Mozilla/5.0'}
# Class to scrape the overview section
class OverviewScraper:
    def __init__(self, url, doc, headers):
        self.url = url
        self.doc = doc
        self.headers = headers
        self.overview_dict = {}
        self.get_overview_section()

    # main method that will perform the scraping
    def get_overview_section(self):

        # get all sections
        overview_sections = self.doc.xpath('//div[starts-with(@class, "embedded-component-container lv product-")]')

        try:
            # description section
            try:
                desc_text = overview_sections[0].text_content().split('(function')[0].split('Description')[1]
            except:
                desc_text = overview_sections[1].text_content().split('(function')[0].split('Description')[1]
        except Exception as e:
            logging.warn(f"{self.url},{e}")
            return []

        try:

            # get all of the list row elements
            list_row_eles = overview_sections[1].xpath('//div[@class="list-row"]')

        except Exception as e :
            logging.warning(f"list of row elements was messed up for overview: {self.url}, {e}")
            return []

        # make a list to hold all features
        features_list = []

        # iterate over all elements
        for ele in list_row_eles:

            # case when header or paragraph exists
            try:

                # get sections of interest
                header = ele.xpath('h4')[0].text
                text = ele.xpath('p')[0].text

                # create header-text dictionary
                header_text_dict = {'header': header, 'description': text}

                # add to list
                features_list.append(header_text_dict)

            # case when neither exists
            except Exception as e:

                logging.warning(f"Either text or header didn't exist for url: {self.url}, {e}")

        # add the results
        self.overview_dict = {
            'description': desc_text,
            'features': features_list
        }


# Class to scrape the specifications section
# class SpecScraper:
#     def __init__(self, url, doc, headers):
#         self.url = url
#         self.doc = doc
#         self.headers = headers
#         self.spec_dict = {}
#         self.get_specs()

#     # get the specifications
#     def get_specs(self):
#         # get to the spec categories page
#         spec_cat_ele = self.doc.xpath('//div[@class="spec-categories"]')

#         # get the section title containers
#         spec_cat_ele = self.doc.xpath('//div[starts-with(@class, "section-title-container")]')


# Class to scrape thumbnails
class ThumbnailScraper:
    def __init__(self, url,   headers):
        self.url = url
        # self.doc = doc
        self.headers = headers
        self.thumbnail_list = []
        try:
            self.get_thumbnails()
        except Exception as e :
            logging.warning(f"something wrong with thumbnails: {self.url}, {e}")
            self.thumbnail_list = []


    # get thumbnails
    def get_thumbnails(self):

        # get starting url
        starting_url = 'https://pisces.bbystatic.com/image2/BestBuy_US/images/products/'

        # get the sku
        sku = self.url.split('skuId=')[-1]

        # get the product number
        prod_num = sku[:-3]

        # get the endpoint
        endpoint = 'd.jpg;maxHeight=54;maxWidth=54'
        endpoint_list = [endpoint,'a.jpg']
        # all endpoints images start with 11 (for whatever reason)
        i = 11

        # add predefined thumbnails
        self.add_predefined_thumbnails(starting_url=starting_url, prod_num=prod_num,
                                       sku=sku, endpoint_list=endpoint_list)

     
        

        while True:

            # get the ith url
            ith_thumbnail_url = starting_url + prod_num + '/' + sku + f'cv{i}' + endpoint

            # make the request
            image_potential = requests.get(ith_thumbnail_url, headers=self.headers)

            # if the history list has at least 1 item, then you've scraped all thumbnails
            if image_potential.content.startswith(b'\x89'):
                break

            # otherwise, add the url to the list
            self.thumbnail_list.append(ith_thumbnail_url)
            # print(image_potential.content)

            # update counter
            i += 1

    # add a thumbnail to the list if it is one of the predefined
    def add_predefined_thumbnails(self, starting_url, prod_num, sku, endpoint_list):

        # get a predefined list of thumbnails
        predefined_patterns = ['_s', '_r', 'l', '_b']

        # iterate over the predefined thumbnail patterns
        for thumbnail_pattern in predefined_patterns:
            for endpoint  in endpoint_list:

                # get the current url
                ith_thumbnail_url = starting_url + prod_num + '/' + sku + thumbnail_pattern + endpoint

                # make the request
                image_potential = requests.get(ith_thumbnail_url, headers=self.headers)

                # if an okay url, then add
                if not image_potential.content.startswith(b'\x89'):
                    # add to list
                    self.thumbnail_list.append(ith_thumbnail_url)



    
class ReviewScraper:
    def __init__(self, url,  headers,is_get_review_in_init=True , review_urls_list=[]):
        self.reviews_list = []
        self.url = url
        # self.doc = doc
        self.headers = headers
        self.review_headers = []
        self.user_info = []
        self.recommendations = []
        self.feedback = []
        self.body_text = []
        self.review_pages = review_urls_list
        if is_get_review_in_init:
            self.get_review_pages()
            self.get_reviews()

    # find the number of review pages
    def get_review_pages(self):

        if len(self.review_pages) > 0:
            return

        # case when there is a review
        try:

            # get the first url
            first_page = self.url.replace('site', 'site/reviews').replace('.p?', '?variant=A&') + '&page=1'

            # get the skuid
            skuid = first_page.split('skuId=')[1].split('&page')[0]

            # go to the url
            content = requests.get(first_page, headers=self.headers).content

            # make a doc object out of it
            first_doc = fromstring(html=content)

            # get the results range
            results_range = first_doc.xpath('//span[@class="message"]')[0].text_content()

            # get total number of reviews
            total_reviews = int(results_range.split(' of ')[1].split('reviews')[0].replace(',', '').strip())

            # find the number of pages
            num_pages = (total_reviews // 20) + 1

            # get all page urls
            rev_page_urls = [f'http://www.bestbuy.com/ugc/v2/reviews?page={i}&pageSize=20&sku={skuid}&sort=BEST_REVIEW&variant=A&verifiedPurchaseOnly=true'
                             for i in range(1, num_pages+1)]

            # get the reviews content
            self.review_pages = rev_page_urls

            # get all review pages
            self.review_pages = [first_page[:-1] + str(i) for i in range(1, num_pages + 1)]

        # case when no reviews are found
        except:
            logging.info(f"The following url doesn't have any reviews: {self.url}")

    # get all reviews for the given page
    def get_reviews(self):

        # exit if there aren't any reviews
        if len(self.review_pages) == 0:
            return

        self._get_reviews()
        # make the requests
        # review_dicts = [requests.get(url=rev_page, headers=self.headers).json() for rev_page in self.review_pages]
        #
        # rev_dicts = [rev_dict['topics'] for rev_dict in review_dicts]
        #
        # self.reviews_list = list(
        #     chain(*rev_dicts)
        # )

    # get all reviews for the given page
    def _get_reviews(self):

        # get the review pages
        self.get_review_pages()

        # exit if there aren't any reviews
        if len(self.review_pages) == 0:
            return

        # iterate over each review page
        review_num=0
        for rev_page in tqdm(self.review_pages):
            if review_num>100:
                break 
            else:
                review_num+=1
            # make the request
            rev_req_content = requests.get(rev_page, headers=self.headers).content

            # make a lxml parser
            doc_rev = fromstring(rev_req_content)

            # get the headers
            (ratings_list, header_list) = self.get_headers(doc_rev)

            # get the user info
            user_info = self.get_user_info(doc_rev)

            # gte recommendation
            recommendations = self.get_recommendations(doc_rev)

            # get feedback info
            (helpful_feedback, unhelpful_feedback) = self.get_feedback(doc_rev)

            # get the bodies
            review_body_text_list = self.get_bodies(doc_rev)
             
            # get the list of thumbnails
            customer_imgs = self.get_review_images(doc_rev, review_body_text_list)

            # zip all lists to make dictionary creation easier
            zipped_reviews = list(zip(user_info, header_list, ratings_list,
                                      recommendations, helpful_feedback,
                                      unhelpful_feedback, review_body_text_list, customer_imgs))

            # review list
            review_list = [
                {'user': rev[0],
                 'header': rev[1],
                 'rating': rev[2],
                 'recommendation': rev[3],
                 'feedback': {
                     'number_helpful': rev[4], 'number_unhelpful': rev[5],
                 },
                 'body': rev[6],
                 'product_images': rev[7]
                 } for rev in zipped_reviews
            ]

            # create a list of dictionaries corresponding to the reviews pulled
            self.reviews_list.extend(review_list)

    # get the review images
    def get_review_images(self, doc_rev, review_body_text_list):

        # dictionary that will help map the profiles with the pictures
        image_header_dict = {}

        # list to hold all image links
        img_link_list = []

        # not all reviews will have images
        try:

            # get the image elements
            gallery = doc_rev.xpath('//ul[@class="carousel gallery-preview"]')

            # iterate over each gallery object ?= each review
            for image_ele in gallery:
                # get the corresponding key (the header)
                image_header = image_ele.xpath('./../*[@class="review-heading"]')[0]
                review_body=image_ele.xpath('./../*[@class="ugc-review-body"]')[0].text_content()
                # find the img link
                img_link_eles = image_ele.xpath('.//li//button//img')

                # get all links
                img_links = [ele.attrib['src'] for ele in img_link_eles]

                # header name
                header_name = image_header.text_content().split('stars')[1]

                # list that corresponds the image and the header list
                image_header_dict[review_body] = img_links

            # list that holds all product links
            # img_link_list = [image_header_dict[header] if header in image_header_dict else []
            #                  for header in header_list]
            img_link_list = [image_header_dict[review_body] if review_body in image_header_dict else []
                             for review_body in review_body_text_list]
        # case when this doesn't work
        except:

            # do nothing
            pass

        # return the image list
        return img_link_list

    # get review headers
    def get_headers(self, doc_rev):

        try:
            # get header and rating elements
            headers_and_ratings = doc_rev.xpath('//div[@class="review-heading"]')

            # separate the ratings from the headings
            ratings = [hr.text_content().split('stars')[0] + 'stars' for hr in headers_and_ratings]
            headers = [hr.text_content().split('stars')[1] for hr in headers_and_ratings]
        except:
            ratings = "N/A"
            headers = "N/A"

        # return both lists
        return (ratings, headers)

    def get_user_info(self, doc_rev):

        # get the elements of all users
        try:
            users = doc_rev.xpath('//div[starts-with(@class,"ugc-author")]')

            # get the user's name
            user_list = [user.text_content() for i, user in enumerate(users) if i % 2 == 0]
        except:
            user_list = []

        # return the users's name
        return user_list

    def get_recommendations(self, doc_rev):

        try:
            # get the recommendation elements
            recommend_eles = doc_rev.xpath('//div[contains(@class, "ugc-recommendation")]')

            # get a list of recommendation texts
            recommend_texts = ['No' if ele.text_content().strip().startswith('No') else 'Yes' for ele in recommend_eles]

        except:
            recommend_texts = []

        # return the list
        return recommend_texts

    def get_feedback(self, doc_rev):

        # case when this works
        try:

            # get the feedback display elements
            feedback_eles = doc_rev.xpath('//div[@class="feedback-display"]')

            # get the number helpful and unhelpful
            feedback_nums = [re.sub("\D", '', ele.text_content()) for ele in feedback_eles]

            # get the number of helpful
            list_helpful = [int(feedback[0]) for feedback in feedback_nums]

            # get the number of unhelpful
            list_unhelpful = [int(feedback[1]) for feedback in feedback_nums]

        # case when it didn't work
        except:

            # debug
            logging.debug(f' Feedback didn"t work for url: {self.url}')
            return ([], [])

        # return the two lists
        return (list_helpful, list_unhelpful)

    # get the body reviews
    def get_bodies(self, doc_rev):
        expect_body ="Read this before you decide to buy this Whirlpool refrigerator or any Whirlpool refrigerator. We bought a new home in May 2020, and purchased this refrigerator. We installed it and moved in July 2020. By June 2021 (less than 1 year) two of the bins completely failed, with the plastic breaking. And the latch on the ice bucket broke rendering the ice machine useless. I opened a ticket with Whirpool and spoke to their customer care supervisors about my experience and they sent me a link to Repair Clinic to order new parts. They also let me know these parts were only covered for 30 days under their \"cosmetic\" warranty. They never once apologized for my experience with their product, nor offered any type of support to resolve the issue. So I'm issuing you fair warning, it's a cheap product. The drawers will break. The latches will break. And there's nothing you can do short of paying large sums of money to order replacement parts to stay on top of the maintenance. Find another brand that manufactures a quality product and stands behind it."
        try:
            # get all body reviews
            body_eles = doc_rev.xpath('//div[@class="ugc-review-body"]')

            # get the body texts
            body_texts=[]
            for body in body_eles:
                if  body.text_content() == expect_body:
                    print("") 
                if not self.is_from_another_product(body):
                    
                    body_texts.append(body.text_content() )
                else:
                    body_texts.append("")
            # body_texts = [body.text_content() for body in body_eles]
        except:
            body_texts = []

        # return the body texts
        return body_texts

    def is_from_another_product(self,body):
        related_product_elements=body.xpath('./../*[@class="body-copy ugc-related-product"]')
        if len(related_product_elements)>0:
            related_product=related_product_elements[0].text_content()
            if "This review is from " in related_product:
                return True 
        return False  


    
class ReviewScraperForDebug:
    def __init__(self, url, review_body, headers, review_urls_list=[]):
        self.reviews_list = []
        self.url = url
        # self.doc = doc
        self.headers = headers
        self.review_headers = []
        self.user_info = []
        self.recommendations = []
        self.feedback = []
        self.body_text = []
        self.review_body_to_search=review_body
        self.review_pages = review_urls_list
        self.get_review_pages()
        self.get_reviews()
        self.wrong_list=[]

    # find the number of review pages
    def get_review_pages(self):

        if len(self.review_pages) > 0:
            return

        # case when there is a review
        try:

            # get the first url
            first_page = self.url.replace('site', 'site/reviews').replace('.p?', '?variant=A&') + '&page=1'

            # get the skuid
            skuid = first_page.split('skuId=')[1].split('&page')[0]

            # go to the url
            content = requests.get(first_page, headers=self.headers).content

            # make a doc object out of it
            first_doc = fromstring(html=content)

            # get the results range
            results_range = first_doc.xpath('//span[@class="message"]')[0].text_content()

            # get total number of reviews
            total_reviews = int(results_range.split(' of ')[1].split('reviews')[0].replace(',', '').strip())

            # find the number of pages
            num_pages = (total_reviews // 20) + 1

            # get all page urls
            rev_page_urls = [f'http://www.bestbuy.com/ugc/v2/reviews?page={i}&pageSize=20&sku={skuid}&sort=BEST_REVIEW&variant=A&verifiedPurchaseOnly=true'
                             for i in range(1, num_pages+1)]

            # get the reviews content
            self.review_pages = rev_page_urls

            # get all review pages
            self.review_pages = [first_page[:-1] + str(i) for i in range(1, num_pages + 1)]

        # case when no reviews are found
        except:
            logging.info(f"The following url doesn't have any reviews: {self.url}")

    # get all reviews for the given page
    def get_reviews(self):

        # exit if there aren't any reviews
        if len(self.review_pages) == 0:
            return

        self._get_reviews()
        # make the requests
        # review_dicts = [requests.get(url=rev_page, headers=self.headers).json() for rev_page in self.review_pages]
        #
        # rev_dicts = [rev_dict['topics'] for rev_dict in review_dicts]
        #
        # self.reviews_list = list(
        #     chain(*rev_dicts)
        # )

    # get all reviews for the given page
    def _get_reviews(self):

        # get the review pages
        self.get_review_pages()

        # exit if there aren't any reviews
        if len(self.review_pages) == 0:
            return

        # iterate over each review page
        review_num=0
        for rev_page in tqdm(self.review_pages):
            if review_num>100:
                break 
            else:
                review_num+=1
            # make the request
            rev_req_content = requests.get(rev_page, headers=self.headers).content

            # make a lxml parser
            doc_rev = fromstring(rev_req_content)

            # get the headers
            (ratings_list, header_list) = self.get_headers(doc_rev)

            # get the user info
            user_info = self.get_user_info(doc_rev)

            # gte recommendation
            recommendations = self.get_recommendations(doc_rev)

            # get feedback info
            (helpful_feedback, unhelpful_feedback) = self.get_feedback(doc_rev)

            # get the bodies
            review_body_text_list = self.get_bodies(doc_rev)
            #TODO
            if self.review_body_to_search in review_body_text_list :
                print("ERROR begin")
                # self.wrong_list.append(rev_page)
            # get the list of thumbnails
            customer_imgs = self.get_review_images(doc_rev, review_body_text_list)

            # zip all lists to make dictionary creation easier
            zipped_reviews = list(zip(user_info, header_list, ratings_list,
                                      recommendations, helpful_feedback,
                                      unhelpful_feedback, review_body_text_list, customer_imgs))

            # review list
            review_list = [
                {'user': rev[0],
                 'header': rev[1],
                 'rating': rev[2],
                 'recommendation': rev[3],
                 'feedback': {
                     'number_helpful': rev[4], 'number_unhelpful': rev[5],
                 },
                 'body': rev[6],
                 'image_url': rev[7]
                 } for rev in zipped_reviews
            ]

            # create a list of dictionaries corresponding to the reviews pulled
            self.reviews_list.extend(review_list)

    # get the review images
    def get_review_images(self, doc_rev, review_body_text_list):

        # dictionary that will help map the profiles with the pictures
        image_header_dict = {}

        # list to hold all image links
        img_link_list = []

        # not all reviews will have images
        try:

            # get the image elements
            gallery = doc_rev.xpath('//ul[@class="carousel gallery-preview"]')

            # iterate over each gallery object ?= each review
            for image_ele in gallery:
                # get the corresponding key (the header)
                image_header = image_ele.xpath('./../*[@class="review-heading"]')[0]
                review_body=image_ele.xpath('./../*[@class="ugc-review-body"]')[0].text_content()
                # find the img link
                img_link_eles = image_ele.xpath('.//li//button//img')

                # get all links
                img_links = [ele.attrib['src'] for ele in img_link_eles]

                # header name
                header_name = image_header.text_content().split('stars')[1]

                # list that corresponds the image and the header list
                image_header_dict[review_body] = img_links

            # list that holds all product links
            # img_link_list = [image_header_dict[header] if header in image_header_dict else []
            #                  for header in header_list]
            img_link_list = [image_header_dict[review_body] if review_body in image_header_dict else []
                             for review_body in review_body_text_list]
        # case when this doesn't work
        except:

            # do nothing
            pass

        # return the image list
        return img_link_list

    # get review headers
    def get_headers(self, doc_rev):

        try:
            # get header and rating elements
            headers_and_ratings = doc_rev.xpath('//div[@class="review-heading"]')

            # separate the ratings from the headings
            ratings = [hr.text_content().split('stars')[0] + 'stars' for hr in headers_and_ratings]
            headers = [hr.text_content().split('stars')[1] for hr in headers_and_ratings]
        except:
            ratings = "N/A"
            headers = "N/A"

        # return both lists
        return (ratings, headers)

    def get_user_info(self, doc_rev):

        # get the elements of all users
        try:
            users = doc_rev.xpath('//div[starts-with(@class,"ugc-author")]')

            # get the user's name
            user_list = [user.text_content() for i, user in enumerate(users) if i % 2 == 0]
        except:
            user_list = []

        # return the users's name
        return user_list

    def get_recommendations(self, doc_rev):

        try:
            # get the recommendation elements
            recommend_eles = doc_rev.xpath('//div[contains(@class, "ugc-recommendation")]')

            # get a list of recommendation texts
            recommend_texts = ['No' if ele.text_content().strip().startswith('No') else 'Yes' for ele in recommend_eles]

        except:
            recommend_texts = []

        # return the list
        return recommend_texts

    def get_feedback(self, doc_rev):

        # case when this works
        try:

            # get the feedback display elements
            feedback_eles = doc_rev.xpath('//div[@class="feedback-display"]')

            # get the number helpful and unhelpful
            feedback_nums = [re.sub("\D", '', ele.text_content()) for ele in feedback_eles]

            # get the number of helpful
            list_helpful = [int(feedback[0]) for feedback in feedback_nums]

            # get the number of unhelpful
            list_unhelpful = [int(feedback[1]) for feedback in feedback_nums]

        # case when it didn't work
        except:

            # debug
            logging.debug(f' Feedback didn"t work for url: {self.url}')
            return ([], [])

        # return the two lists
        return (list_helpful, list_unhelpful)

    # get the body reviews
    def get_bodies(self, doc_rev):

        try:
            # get all body reviews
            body_eles = doc_rev.xpath('//div[@class="ugc-review-body"]')

            # get the body texts
            body_texts = [body.text_content() for body in body_eles]
        except:
            body_texts = []

        # return the body texts
        return body_texts

import json 

class ReviewScraperWithoutReviewOfRelatedProduct(ReviewScraper) :
    def __init__(self, url, review_body, headers ):
        super().__init__( url,  headers,False )
        self.reviews_list = []
        self.product_url = url
        self.expect_review_body=review_body
        self.headers = headers
     
 
        self.get_review_pages()
        self.get_reviews()
        
    def get_reviews(self):
        # get the review pages
        self.get_review_pages()

        # exit if there aren't any reviews
        if len(self.review_pages) == 0:
            return

        # iterate over each review page
        review_num=0
        for rev_page in tqdm(self.review_pages):
            if review_num>100:
                break 
            else:
                review_num+=1
            # make the request
            # review_body_html_encoded=urllib.parse.quote(self.expect_review_body)
            
            # rev_page=rev_page+"&searchText="+review_body_html_encoded +"&sort=BEST_MATCH"
            rev_req_content = requests.get(rev_page, headers=self.headers).content

            # make a lxml parser
            doc_rev = fromstring(rev_req_content)

            # get the headers
            (ratings_list, header_list) = self.get_headers(doc_rev)

            # get the user info
            user_info = self.get_user_info(doc_rev)

            # gte recommendation
            recommendations = self.get_recommendations(doc_rev)

            # get feedback info
            (helpful_feedback, unhelpful_feedback) = self.get_feedback(doc_rev)

            # get the bodies
            review_body_text_list = self.get_bodies(doc_rev)
             
            # get the list of thumbnails
            customer_imgs = self.get_review_images(doc_rev, review_body_text_list)

            # zip all lists to make dictionary creation easier
            zipped_reviews = list(zip(user_info, header_list, ratings_list,
                                      recommendations, helpful_feedback,
                                      unhelpful_feedback, review_body_text_list, customer_imgs))

            # review list
            review_list=[]
            for idx,rev in enumerate(zipped_reviews):
                if review_body_text_list[idx]!="":
                    review_list.append(
                        {'user': rev[0],
                        'header': rev[1],
                        'rating': rev[2],
                        'recommendation': rev[3],
                        'feedback': {
                            'number_helpful': rev[4], 'number_unhelpful': rev[5],
                        },
                        'body': rev[6],
                        'product_images': rev[7]
                        } ) 
                    
            

            # create a list of dictionaries corresponding to the reviews pulled
            self.reviews_list.extend(review_list) 
         
import re
import unidecode 

 
import html
def unescape(s):
    return html.unescape(s)
   

def clean(s):
    s=unescape(s)
    s=unidecode.unidecode(s)
    s =re.sub(r'(a-zA-Z0-9 \t\n\r\f\v_)+', '', s)
    
    return s 
class SearchReviewScraper :
    def __init__(self, url, review_body, product_name,headers,review_id ,review_header):
        # super().__init__( url,  headers)
        self.reviews_list = []
        self.product_url = url
        self.review_body=review_body
        self.headers = headers
        self.product_name_to_search=product_name
        self.review_id=review_id
        self.review_header=review_header
 
        self.get_review_pages()
        
    def get_reviews(self):
        pass 
        
    def search_by_text(self,url,review_body):
        content = requests.get(url, headers=self.headers).content
        response_json=json.loads(content)
        review_image_url_list=[]
            
        is_valid_review=False 
        is_found=False 
        for one_review in response_json["topics"]:
            review_body_in_html=one_review["text"]
            if clean(remove_html_tags(review_body_in_html))[:150]==clean(review_body)[:150]:
                is_found=True 
                for photo in one_review["photos"]:
                    if photo["piscesUrl"] is not None:
                        review_image_url_list.append(photo["piscesUrl"])
                    elif  photo["normalUrl"] is not None:
                        review_image_url_list.append(photo["normalUrl"])
                    
                    
                if "productDetails" in one_review:
                    is_valid_review=False  
                    # product_name=one_review["productDetails"]["name"]
                    
                    # if product_name==self.product_name_to_search:
                    #     is_valid_review=True 
                            
                    # else:
                    #     is_valid_review=False  
                else:
                
                    is_valid_review=True 
                break  
        return is_found,is_valid_review,review_image_url_list
    
     
        
    def get_review_pages(self):
        try:
            prefix_url="https://www.bestbuy.com/ugc/v2/reviews?pageSize=20"
            skuid= self.product_url.split('skuId=')[1] 
            review_body=self.review_body
            review_body_html_encoded=urllib.parse.quote(review_body[:150])
            url=prefix_url+"&searchText="+review_body_html_encoded+"&sku="+skuid+"&sort=BEST_MATCH&variant=A"
            
            is_found,is_valid_review,review_image_url_list=self.search_by_text( url,review_body)
            if not is_found:
                query_text=self.review_header 
                review_body_html_encoded=urllib.parse.quote(query_text[:150])
                header_url=prefix_url+"&searchText="+review_body_html_encoded+"&sku="+skuid+"&sort=BEST_MATCH&variant=A"
                is_found,is_valid_review,review_image_url_list=self.search_by_text( header_url,review_body)
                if not is_found:
                    for i in range(2,11):
                        next_page_url=url+f"&page={str(i)}"
                        is_found,is_valid_review,review_image_url_list=self.search_by_text( next_page_url,review_body)
                        if is_found:
                            break 
            # go to the url
            
                            
            if not  is_found :
                logging.info(f"not find review for {self.review_id} ")
            if is_found and is_valid_review:
                review_list = [
                    {'user':None ,
                    'header': None,
                    'rating': None,
                    'recommendation': None,
                    'feedback': {
                        'number_helpful': None, 'number_unhelpful': None,
                    },
                    'body': review_body,
                    'image_url': review_image_url_list
                    } 
                ]
            else:
                review_list= [
                    {'user':None ,
                    'header': None,
                    'rating': None,
                    'recommendation': None,
                    'feedback': {
                        'number_helpful': None, 'number_unhelpful': None,
                    },
                    'body': "",
                    'image_url': []
                    } 
                ]

            # create a list of dictionaries corresponding to the reviews pulled
            self.reviews_list.extend(review_list)
            # make a doc object out of it
            # first_doc = fromstring(html=content)
            # print(first_doc)
            # get the results range
            # results_range = first_doc.xpath('//span[@class="message"]')[0].text_content()
        except Exception as e :
            logging.info(f"The following url doesn't have any reviews: {self.product_url} {e}")
class SpecScraper:
    def __init__(self, url, driver):
        self.url = url
        self.driver=driver
        self.wdwait = WebDriverWait(driver, 10)
        self.spec_list_dict = self.get_specs()

    # get the specifications
    # get the description information
    def get_specs(self):

        # list that will hold all of the dictionaries
        spec_list_dict = []
        resp = requests.get(self.url, headers=headers)
        if resp.status_code != 404:
            # case when the specification name is the one listed above
            try:

                # now try to get the specifications
                spec_element = self.wdwait.until(EC.element_to_be_clickable((By.CLASS_NAME, 'shop-specifications')))

            # case when the specification name is slightly off
            except:

                # determine if the spec format is in a different layout
                try:

                    # now try to get the specifications
                    spec_element = self.wdwait.until(EC.element_to_be_clickable((By.CLASS_NAME, 'shop-mp-specifications')))

                # case when there aren't any specifications
                except Exception as e:
                    logging.warn(f"{self.url },{e}")
                    # return empty dictionary
                    return []
        else:
            return []

        # now find specifications button
        spec_element.find_element(By.TAG_NAME, 'button').click()

        # find the spec categories
        spec_cat_ele = spec_element.find_element(By.CLASS_NAME, 'spec-categories')

        # explicit wait
      
        # time.sleep(2)

        # find the category wrapper
        wait_time=0
        wait_time_once=0.05
        while wait_time<2:
            try:
                spec_cat_wrappers = spec_cat_ele.find_elements(By.CLASS_NAME, 'category-wrapper')
                if len(spec_cat_wrappers)>0:
                    break 
            except:
                wait_time+=wait_time_once
                time.sleep(wait_time_once)
               
                
        # spec_cat_wrappers = spec_cat_ele.find_elements(By.TAG_NAME, 'div')
        print(len(spec_cat_wrappers))

        # iterate over all spec categories
        
        try:
            for spec_wrapper in spec_cat_wrappers:

                # case when not at the bottom
                try:

                    # find the description element
                    prod_specs = spec_wrapper.find_element(By.CLASS_NAME, 'specifications-list')

                # case when at the bottom
                except:
                    continue

                # get list items
                spec_items = prod_specs.find_elements(By.CLASS_NAME, 'list-item')

                # iterate over each spec item
                for spec_item in spec_items:
                    # split the text by new line
                    spec_text = spec_item.text.split('\n')

                    # the category will be on the left and the value will be on the right
                    specification = spec_text[0]
                    value = spec_text[1]

                    # create a dictionary mapping the specification to the value
                    spec_val_dict = {
                        'specification': specification,
                        'value': value
                    }

                    # add this dictionary to a list of dictionaries
                    spec_list_dict.append(spec_val_dict)
        except Exception as e:
            logging.warn(f"{self.url },{e}")
            return []

        # return the list of dictionaries
        return spec_list_dict
    
    
def remove_html_tags(text):
    """Remove html tags from a string"""
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)
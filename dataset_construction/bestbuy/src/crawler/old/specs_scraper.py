from lxml.html import fromstring
from tqdm import tqdm
import requests
import logging
import time
import re
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
headers = {'User-Agent': 'Mozilla/5.0'}
# Class to scrape the specifications section


# Class to scrape the overview section
class OverviewScraper():
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

        # when the "skuID" is not in the url, that means the overview section is already open
        if 'skuid' not in self.url.lower():
            pass


        # description section
        try:
            desc_text = overview_sections[0].text_content().split('(function')[0].split('Description')[1]
        except:
            desc_text = overview_sections[1].text_content().split('(function')[0].split('Description')[1]

        # get all of the list row elements
        list_row_eles = overview_sections[1].xpath('//div[@class="list-row"]')

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
                header_text_dict = {'header': header, 'description':text}

                # add to list
                features_list.append(header_text_dict)

            # case when neither exists
            except Exception as e:
                logging.warning(f"Either text or header didn't exist for url: {self.url} {e}")

        # add the results
        self.overview_dict = {
            'description': desc_text,
            'features': features_list
        }


# Class to scrape thumbnails
class ThumbnailScraper:
    def __init__(self, url, doc, headers):
        self.url = url
        self.doc = doc
        self.headers = headers
        self.thumbnail_list = []
        self.get_thumbnails()

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

        # all endpoints images_total start with 11 (for whatever reason)
        i = 11

        # add predefined thumbnails
        self.add_predefined_thumbnails(starting_url=starting_url, prod_num=prod_num,
                                       sku=sku, endpoint=endpoint)

        # iterate until there isn't any thumbnail images_total
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
    def add_predefined_thumbnails(self, starting_url, prod_num, sku, endpoint):

        # get a predefined list of thumbnails
        predefined_patterns = ['_s', '_r', 'l', '_b']

        # iterate over the predefined thumbnail patterns
        for thumbnail_pattern in predefined_patterns:

            # get the current url
            ith_thumbnail_url = starting_url + prod_num + '/' + sku + thumbnail_pattern + endpoint

            # make the request
            image_potential = requests.get(ith_thumbnail_url, headers=self.headers)

            # if an okay url, then add
            if not image_potential.content.startswith(b'\x89'):

                # add to list
                self.thumbnail_list.append(ith_thumbnail_url)

class ReviewScraper:
    def __init__(self, url, doc, headers):
        self.url = url
        self.doc = doc
        self.headers = headers
        self.review_headers = []
        self.user_info = []
        self.recommendations = []
        self.feedback = []
        self.body_text = []
        self.review_pages = []
        self.reviews_list = []
        self.get_reviews()

    # find the number of review pages
    def get_review_pages(self):

        # case when there is a review
        try:

            # get the first url
            first_page = self.url.replace('site', 'site/reviews').replace('.p?', '?variant=A&') + '&page=1'

            # go to the url
            content = requests.get(first_page, headers=self.headers).content

            # make a doc object out of it
            first_doc = fromstring(html=content)

            # get the results range
            results_range = first_doc.xpath('//span[@class="message"]')[0].text_content()

            # get total number of reviews
            total_reviews = int(results_range.split(' of ')[1].split('reviews')[0].replace(',','').strip())

            # find the number of pages
            num_pages = (total_reviews // 20) + 1

            # get all review pages
            self.review_pages = [first_page[:-1] + str(i) for i in range(1, num_pages + 1)]

        # case when no reviews are found
        except Exception as e:
            logging.warning(f"The following url doesn't have any reviews: {self.url}, {e}")


    # get all reviews for the given page
    def get_reviews(self):

        # get the review pages
        self.get_review_pages()

        # exit if there aren't any reviews
        if len(self.review_pages) == 0:
            return

        # iterate over each review page
        for rev_page in tqdm(self.review_pages):

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
            body_texts = self.get_bodies(doc_rev)

            # get the list of thumbnails
            customer_imgs = self.get_review_images(doc_rev, header_list)

            # zip all lists to make dictionary creation easier
            zipped_reviews = list(zip(user_info, header_list, ratings_list,
                                 recommendations, helpful_feedback,
                                 unhelpful_feedback, body_texts, customer_imgs))


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

    def _get_reviews(self):
        pass

    # get the review images_total
    def get_review_images(self, doc_rev, header_list):

        # dictionary that will help map the profiles with the pictures
        image_header_dict = {}

        # list to hold all image links
        img_link_list = []

        # not all reviews will have images_total
        try:

            # get the image elements
            gallery = doc_rev.xpath('//ul[@class="carousel gallery-preview"]')

            # iterate over each gallery object ?= each review
            for image_ele in gallery:

                # get the corresponding key (the header)
                image_header = image_ele.xpath('./../*[@class="review-heading"]')[0]

                # find the img link
                img_link_eles = image_ele.xpath('.//li//button//img')

                # get all links
                img_links = [ele.attrib['src'] for ele in img_link_eles]

                # header name
                header_name = image_header.text_content().split('stars')[1]

                # list that corresponds the image and the header list
                image_header_dict[header_name] = img_links

            # list that holds all product links
            img_link_list = [image_header_dict[header] if header in image_header_dict else []
                             for header in header_list]

        # case when this doesn't work
        except Exception as e:
            logging.warn(f"{self.url },{e}")
            # do nothing
            pass

        # return the image list
        return img_link_list



    # get review headers
    def get_headers(self, doc_rev):

        # get header and rating elements
        headers_and_ratings = doc_rev.xpath('//div[@class="review-heading"]')

        # separate the ratings from the headings
        ratings = [hr.text_content().split('stars')[0] + 'stars' for hr in headers_and_ratings]
        headers = [hr.text_content().split('stars')[1] for hr in headers_and_ratings]

        # return both lists
        return (ratings, headers)

    def get_user_info(self, doc_rev):

        # get the elements of all users
        users = doc_rev.xpath('//div[starts-with(@class,"ugc-author")]')

        # get the user's name
        user_list = [user.text_content() for i,user in enumerate(users) if i%2==0]

        # return the users's name
        return user_list

    def get_recommendations(self, doc_rev):

        # get the recommendation elements
        recommend_eles = doc_rev.xpath('//div[contains(@class, "ugc-recommendation")]')

        # get a list of recommendation texts
        recommend_texts = ['No' if ele.text_content().strip().startswith('No') else 'Yes' for ele in recommend_eles]

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
        except Exception as e :

            # debug
            logging.warning(f' Feedback didn"t work for url: {self.url},{e}')
            return ([], [])

        # return the two lists
        return (list_helpful, list_unhelpful)


    # get the body reviews
    def get_bodies(self, doc_rev):

        # get all body reviews
        body_eles = doc_rev.xpath('//div[@class="ugc-review-body"]')

        # get the body texts
        body_texts = [body.text_content() for body in body_eles]

        # return the body texts
        return body_texts

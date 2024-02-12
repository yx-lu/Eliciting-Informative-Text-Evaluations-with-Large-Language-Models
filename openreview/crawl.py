import requests
import json
import csv

BASE_URL = "https://api.openreview.net/notes?invitation=ICLR.cc/2020/Conference/-/Blind_Submission&details=directReplies"

def get_iclr2020_reviews(paper_info_filename = None, review_info_filename = None, decision_info_filename = None):
    offset = 0
    limit = 10  # number of paper per request
    max_limit = 10 # float('inf')
    paper_list = []
    review_list = []
    decision_list = []

    while offset + limit <= max_limit:
        print(f'Scraping review data: {offset}/{max_limit}')
        response = requests.get(BASE_URL, params={"offset": offset, "limit": limit})
        if response.status_code != 200:
            print("Error:", response.status_code)
            break

        data = response.json()
        if not data or 'notes' not in data or not data['notes']:
            break

        for note in data['notes']:
            paper = {}

            paper['uid'] = note['id']
            paper['number'] = note['number']
            paper['title'] = note['content']['title']
            paper['authors'] = note['content']['authors']
            paper['abstract'] = note['content']['abstract']
            paper['pdf'] = note['content']['pdf']
            paper['keywords'] = note['content']['keywords']

            if 'details' in note and 'directReplies' in note['details']:
                paper_list.append(paper)
                for directReplies in note['details']['directReplies']:

                    print(directReplies)
                    input()

                    if directReplies['invitation'].endswith("Official_Review"):
                        
                        review = {}
                        review['uid'] = directReplies['id']
                        review['paper_uid'] = paper['uid']
                        review['paper_title'] = paper['title']
                        review.update(directReplies['content'])
                        
                        review_list.append(review)

                    if directReplies['invitation'].endswith("Decision"):

                        decision = {}
                        decision['uid'] = directReplies['id']
                        decision['paper_uid'] = paper['uid']
                        decision['paper_title'] = paper['title']
                        decision.update(directReplies['content'])
                        
                        decision_list.append(decision)

        offset += limit

    if paper_info_filename != None:
        with open(paper_info_filename, 'w', newline='') as paper_csv:
            writer = csv.DictWriter(paper_csv, fieldnames=paper_list[0].keys())
        
            writer.writeheader()
            
            for row in paper_list:
                writer.writerow(row)
    
    if review_info_filename != None:
        with open(review_info_filename, 'w', newline='') as review_csv:
            writer = csv.DictWriter(review_csv, fieldnames=review_list[0].keys())
        
            writer.writeheader()
            
            for row in review_list:
                writer.writerow(row)

    if decision_info_filename != None:
        with open(decision_info_filename, 'w', newline='') as decision_csv:
            writer = csv.DictWriter(decision_csv, fieldnames=decision_list[0].keys())
        
            writer.writeheader()
            
            for row in decision_list:
                writer.writerow(row)

    return paper_list, review_list, decision_list

if __name__ == "__main__":
    get_iclr2020_reviews(paper_info_filename='ICLR2020paper_10.csv',review_info_filename='ICLR2020review_10.csv',decision_info_filename='ICLR2020decision_10.csv')

    print("Finished scraping ICLR 2020 reviews!")

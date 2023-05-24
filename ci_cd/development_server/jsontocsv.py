import json
import csv
from time import sleep
import requests

from github import Github as GitHub, RateLimitExceededException

OUTPUT_CSV_FILE = "repositories.csv"
json_path = "repos.json"

f = open(json_path)


def getUrl(url):
    """ Given a URL it returns its body """
    response = requests.get(url)
    return response.json()


data = json.load(f)

csv_file = open(OUTPUT_CSV_FILE, 'w')
repositories = csv.writer(csv_file, delimiter=',')
repositories.writerow(['name', 'size', 'star_count', 'folk_counts', 'open_issue', 'has_wiki',
                      'Java', 'Python', 'C', 'JavaScript', 'Ruby', 'Objective_C', 'Other_language', 'wathcer_counts'])
countOfRepositories = 0
count = 0
for item in data:
    name = item['name']
    size = item['size']
    star_counts = item['stargazers_count']
    open_issues_count = item['open_issues_count']
    fork_counts = item['forks_count']
    wiki_bool = item['has_wiki']
    follower_url = item['subscribers_url']
    watch_counts = item['watchers_count']
    # if count < 30:
    #     subscribers = getUrl(follower_url)
    #     subscribers_counts = len(subscribers)
    #     print(subscribers_counts)
    # else:
    #     print("Hit rate limit, sleeping 60 seconds...")
    #     count = 0
    #     sleep(60)
    has_wiki = 0
    Java = 0
    Python = 0
    C = 0
    JavaScript = 0
    Ruby = 0
    Objective_C = 0
    other_language = 0
    if wiki_bool:
        has_wiki = 1
    language = item['language']
    if language == 'Java':
        Java = 1
    elif language == 'Python':
        Python = 1
    elif language == 'C':
        C = 1
    elif JavaScript == 'JavaScript':
        JavaScript = 1
    elif Ruby == 'Ruby':
        Ruby = 1
    elif Objective_C == 'Objective_C':
        Objective_C = 1
    else:
        other_language = 1
    repositories.writerow([name, size, star_counts, fork_counts, open_issues_count,
                           has_wiki, Java, Python, C, JavaScript, Ruby, Objective_C, other_language, watch_counts])
    countOfRepositories = countOfRepositories + 1
    count += 1

print("DONE! " + str(countOfRepositories) +
      " repositories have been processed.")
csv_file.close()

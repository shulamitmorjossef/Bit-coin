# import requests
# from bs4 import BeautifulSoup
# import time
#
# def get_all_users(url='https://bitcoin-otc.com/viewratings.php'):
#     response = requests.get(url)
#     print(f"Main page status code: {response.status_code}")
#     soup = BeautifulSoup(response.text, 'html.parser')
#
#     tables = soup.find_all('table')
#     user_table = None
#     for table in tables:
#         headers_row = table.find_all('th')
#         headers_text = [th.text.strip().lower() for th in headers_row]
#         if 'id' in headers_text and 'nick' in headers_text and any('first rated' in h for h in headers_text):
#             user_table = table
#             break
#
#     if not user_table:
#         print("User table not found!")
#         return []
#
#     users = []
#     for row in user_table.find_all('tr')[1:]:
#         cols = row.find_all('td')
#         if len(cols) >= 2:
#             user_nick = cols[1].text.strip()
#             users.append(user_nick)
#     return users
#
# def get_user_ratings(user_nick):
#     url_user = f'https://bitcoin-otc.com/viewratingdetail.php?nick={user_nick}&sign=ANY&type=RECV'
#     response = requests.get(url_user)
#     soup = BeautifulSoup(response.text, 'html.parser')
#
#     tables = soup.find_all('table')
#     ratings_table = None
#     for table in tables:
#         headers = [th.text.strip().lower() for th in table.find_all('th')]
#         if 'rater nick' in headers and 'rating' in headers:
#             ratings_table = table
#             break
#
#     if not ratings_table:
#         return []
#
#     ratings = []
#     for row in ratings_table.find_all('tr')[1:]:
#         cols = row.find_all('td')
#         if len(cols) >= 7:
#             rater_nick = cols[1].text.strip()
#             rating = cols[5].text.strip()
#             notes = cols[6].text.strip()
#             ratings.append((rater_nick, rating, notes))
#     return ratings
#
# def main():
#     print("Fetching all users...")
#     users = get_all_users()
#     print(f"Total users found: {len(users)}")
#
#     with_comment = 0
#     without_comment = 0
#
#     for idx, user_nick in enumerate(users):
#         print(f"\n[{idx+1}/{len(users)}] Checking ratings for: {user_nick}")
#         try:
#             ratings = get_user_ratings(user_nick)
#         except Exception as e:
#             print(f"Error fetching {user_nick}: {e}")
#             continue
#
#         for rater, rating, note in ratings:
#             if rating == '10' or rating == '-10':
#                 comment = note if note.strip() else "No comment"
#                 print(f"{rater} rated {user_nick} with {rating}. Comment: {comment}")
#                 if note.strip():
#                     with_comment += 1
#                 else:
#                     without_comment += 1
#
#         time.sleep(1)  # respectful delay to avoid overwhelming the server
#
#     print("\nSummary:")
#     print(f"Ratings with comments: {with_comment}")
#     print(f"Ratings without comments: {without_comment}")
#
# if __name__ == '__main__':
#     main()

import requests
from bs4 import BeautifulSoup
import time

def safe_get(url, retries=3, delay=2):
    for i in range(retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response
            else:
                print(f"Non-200 response ({response.status_code}) for {url}")
        except requests.exceptions.RequestException as e:
            print(f"Request error on attempt {i+1}/{retries} for {url}: {e}")
        time.sleep(delay)
    return None

def get_all_users(url='https://bitcoin-otc.com/viewratings.php'):
    response = safe_get(url)
    if not response:
        print("Failed to fetch main user list.")
        return []

    print(f"Main page status code: {response.status_code}")
    soup = BeautifulSoup(response.text, 'html.parser')

    tables = soup.find_all('table')
    user_table = None
    for table in tables:
        headers_row = table.find_all('th')
        headers_text = [th.text.strip().lower() for th in headers_row]
        if 'id' in headers_text and 'nick' in headers_text and any('first rated' in h for h in headers_text):
            user_table = table
            break

    if not user_table:
        print("User table not found!")
        return []

    users = []
    for row in user_table.find_all('tr')[1:]:
        cols = row.find_all('td')
        if len(cols) >= 2:
            user_nick = cols[1].text.strip()
            users.append(user_nick)
    return users

def get_user_ratings(user_nick):
    url_user = f'https://bitcoin-otc.com/viewratingdetail.php?nick={user_nick}&sign=ANY&type=RECV'
    response = safe_get(url_user)
    if not response:
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    tables = soup.find_all('table')
    ratings_table = None
    for table in tables:
        headers = [th.text.strip().lower() for th in table.find_all('th')]
        if 'rater nick' in headers and 'rating' in headers:
            ratings_table = table
            break

    if not ratings_table:
        return []

    ratings = []
    for row in ratings_table.find_all('tr')[1:]:
        cols = row.find_all('td')
        if len(cols) >= 7:
            rater_nick = cols[1].text.strip()
            rating = cols[5].text.strip()
            notes = cols[6].text.strip()
            ratings.append((rater_nick, rating, notes))
    return ratings

def main():
    print("Fetching all users...")
    users = get_all_users()
    print(f"Total users found: {len(users)}")

    with_comment = 0
    without_comment = 0

    for idx, user_nick in enumerate(users):
        print(f"\n[{idx+1}/{len(users)}] Checking ratings for: {user_nick}")
        try:
            ratings = get_user_ratings(user_nick)
        except Exception as e:
            print(f"Error fetching {user_nick}: {e}")
            continue

        for rater, rating, note in ratings:
            if rating == '10' or rating == '-10':
                comment = note if note.strip() else "No comment"
                print(f"{rater} rated {user_nick} with {rating}. Comment: {comment}")
                if note.strip():
                    with_comment += 1
                else:
                    without_comment += 1

        time.sleep(1)  # respectful delay

    print("\nSummary:")
    print(f"Ratings with comments: {with_comment}")
    print(f"Ratings without comments: {without_comment}")

if __name__ == '__main__':
    main()

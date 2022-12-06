from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

import time

def launch_search(place_name, path):
    driver = webdriver.Firefox()
    driver.get("https://www.geoportail.gouv.fr/")
    time.sleep(2)

    driver.fullscreen_window()

    elem = driver.find_element(By.XPATH, "//*[@id='main-search-input']")
    elem.send_keys(place_name)
    time.sleep(2)
    elem.send_keys(Keys.RETURN)

    time.sleep(2)
    passer_accueil = driver.find_element(By.XPATH, '//*[@id="help-layer"]')
    passer_accueil.click()

    time.sleep(3)
    bouton_jaune = driver.find_element(By.XPATH,'//*[@id="geolocation-marker"]')#'//*[@id="geolocation-marker-close"]'
    bouton_jaune.click()

    time.sleep(5)
    #driver.find_element(By.XPATH, '/html/body/div[2]/div[5]/section/div[15]/div[3]/div[4]') :
    close_bubble = driver.find_element(By.CSS_SELECTOR, '.gp-styling-button')
    close_bubble.click()

    #bouton_zoom = driver.find_element(By.XPATH, '//*[@id="zoom-in"]')
    #bouton_zoom.click()

    time.sleep(5)
    driver.get_screenshot_as_file(path)#'/home/mati/code/Mati-Gonc/Earth-Project/screenshots/test1.png')
    driver.close()
    return None

#close = str(input('wannaclose ?'))
#if close == 'y' : driver.close()

#echelle = driver.find_element(By.ID, 'numeric-scale-denominator-input')
#print(echelle)

#bouton_zoom =

#bouton = driver.find_element(By.XPATH, '')


#bouton_ok = driver.find_element(By.XPATH, '/html/body/div[2]/div[5]/section/form/div[2]/div[1]/p[1]')
#bouton_ok.click()

""" bouton = driver.find_element(By.XPATH, '/html/body/c-wiz/div/div/div/div[2]/div[1]/div[3]/div[1]/div[1]/form[2]/div/div/button/span')
bouton.click()

#assert "google/maps/" in driver.title

elem = driver.find_element(By.XPATH, '//*[@id="searchboxinput"]')
#elem.clear()
elem.send_keys("Brioude")
elem.send_keys(Keys.RETURN)

bouton = driver.find_element(By.XPATH, '/html/body/div[3]/div[9]/div[9]/div/div/div[2]/button/img')
bouton.click()

bouton = driver.find_element(By.XPATH,'/html/body/div[3]/div[9]/div[9]/div/div/div[2]/button/img')
bouton.click()


#assert "No results found." not in driver.page_source
#driver.close()
 """

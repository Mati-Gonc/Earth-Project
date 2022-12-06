from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains

import time

driver = webdriver.Firefox()
driver.get("https://www.google.com/maps/@48.8656,2.3789893,14z")

bouton = driver.find_element(By.XPATH, '/html/body/c-wiz/div/div/div/div[2]/div[1]/div[3]/div[1]/div[1]/form[2]/div/div/button/span')
bouton.click()

time.sleep(2)
#assert "google/maps/" in driver.title

#elem = driver.find_element(By.ID, "XmI62e")
elem = driver.find_element(By.XPATH, '//*[@id="searchboxinput"]')
#elem.clear()
elem.send_keys("Brioude")
elem.send_keys(Keys.RETURN)


actions = ActionChains(driver)
actions.move_to_element(driver.find_element(By.XPATH, '/html/body/div[3]/div[9]/div[23]/div[5]/div/div[2]/button'))
actions.click(driver.find_element(By.XPATH, '/html/body/div[3]/div[9]/div[23]/div[7]/div'))
actions.perform()


#bouton = driver.find_element(By.XPATH, '/html/body/div[3]/div[9]/div[9]/div/div/div[2]/button/img')
#bouton.click()

#bouton = driver.find_element(By.XPATH,'/html/body/div[3]/div[9]/div[9]/div/div/div[2]/button/img')
#bouton.click()


#assert "No results found." not in driver.page_source
#driver.close()

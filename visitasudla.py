from selenium import webdriver   #pip install selenium
from selenium.webdriver.common.by import By
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
import time
import datetime
import random
import pyautogui

file = open(r'hist.txt','a')


#lista para acceder a las carreras de manera aleatoria, son 9 carreras que se muestran en la pagina de FICA
list_rand = [1, 2, 3, 4, 5, 6, 7, 8, 9] #sacar automatico-------------

#funciones que generan tiempos aleatorios, tiempo en decimal y tiempos enteros
def tiempos_decimal (rango_ini,rango_fin):
    return random.uniform(rango_ini,rango_fin)

def tiempos_enteros (rango_ini,rango_fin):
    return random.randint(rango_ini,rango_fin)

#rango entero int es el numero de veces que se va a hacer scroll en la pantalla y rango dec es la posicion donde se va a hacer scroll
def tiempos_scroll (rango_ini_int,rango_fin_int,rango_ini_dec,rango_fin_dec):
    for n in range(tiempos_enteros(rango_ini_int,rango_fin_int)):
        t = tiempos_decimal(rango_ini_dec,rango_fin_dec)
        driver.execute_script(f"window.scrollBy(0,{t})","")
        time.sleep(tiempos_decimal(5,25))
    driver.execute_script("window.scrollBy(0,0)","")
    return

espera_inicio = tiempos_decimal(10,30)
print(f'El programa iniciara en {espera_inicio} sg')
time.sleep(espera_inicio)

#vamos a medir el tiempo de la ejcucion
tiempo_inicio = time.time()


#Inicio del bot
option = webdriver.EdgeOptions()
option.add_argument('--start-maximized')
#option.add_argument('--disable-infobars')
s = Service('C:/Users/inteligencia/Desktop/webdriver/msedgedriver.exe')
driver = webdriver.Edge(service=s,options=option)
driver.get("https://www.udla.edu.ec/admisiones-udla/")
x = tiempos_decimal(8,10)
time.sleep(x)
pyautogui.click(300, 500)
#print("///////////////////////////////////////// ",x)
time.sleep(x)

try:
    tiempos_scroll(1,10,300,2000)#significa que se va a scrollear de 1 a 5 veces en una posicion aleatoria entre 500 y 2000 en la pantalla
    pre =  driver.find_element(By.XPATH, '//*[@id="mega-menu-item-23"]/a/span') #menu de facultades
    pre.click()
    pre.click()
    time.sleep(tiempos_decimal(10,18))  
    fica =driver.find_element(By.XPATH,'//*[@id="mega-menu-item-7942"]/a')#Facultad FICA---sacar todos los ids de facultades
    fica.click() 
    time.sleep(tiempos_decimal(10,20))
    #driver.execute_script("window.scrollBy(0,1100)","")
    #se escoge un numero entre 1 - 9 que corresponden a las carreras dentro de FICA
    num_rand = random.choice(list_rand)
    #time.sleep(5)

    #Aleatorio entre carreras fica, la condicion es para hacer scroll ya que se debe tener el boton visible antes de hacer clic
    if num_rand == 1 or num_rand == 2 or num_rand == 3:
        driver.execute_script("window.scrollBy(0,700)","")
        time.sleep(tiempos_decimal(8,16))
        carrera =driver.find_element(By.XPATH,'/html/body/div[1]/div/main/div/section[3]/div/div/div/div[2]/div/div[1]/div/div['+str(num_rand)+']/div/div/div[4]/a')
        time.sleep(tiempos_decimal(5,12))
        carrera.click()
    elif num_rand == 4 or num_rand == 5 or num_rand == 6:
        driver.execute_script("window.scrollBy(0,1100)","")
        time.sleep(tiempos_decimal(8,16))
        carrera =driver.find_element(By.XPATH,'/html/body/div[1]/div/main/div/section[3]/div/div/div/div[2]/div/div[1]/div/div['+str(num_rand)+']/div/div/div[4]/a')
        time.sleep(tiempos_decimal(5,12))
        carrera.click()
    elif num_rand == 7 or num_rand == 8 or num_rand == 9:
        driver.execute_script("window.scrollBy(0,1100)","")
        time.sleep(tiempos_decimal(8,16))
        carrera =driver.find_element(By.XPATH,'/html/body/div[1]/div/main/div/section[3]/div/div/div/div[2]/div/div[1]/div/div['+str(num_rand)+']/div/div/div[4]/a')
        time.sleep(tiempos_decimal(5,12))
        carrera.click()
    else:
        print("Error al escoger carrera")
        
    time.sleep(tiempos_decimal(10,18))
    
    #finalmente al elegir una carrera navega en la pagina
    tiempos_scroll(5,10,100,2000)
    
finally:
    # Cerrar el navegador al finalizar
    driver.quit()
    #Fin del tiempo para medir la ejecucion del bot
    tiempo_fin = time.time()
    tiempo_ejecucion = tiempo_fin - tiempo_inicio
    #print(f"El duración del bot fue de:  {tiempo_ejecucion}.")
    minutos, segundos = divmod(tiempo_ejecucion, 60)
    file.write(f'{datetime.datetime.now()} - Duracion bot: {tiempo_ejecucion}, {minutos} : {segundos} \n')

# import pygoogle_image
# from pygoogle_image import image as pi
# poses = ['PADOTTHANASANA', 'PARVATASANA', 'ARDHA TITALI ASANA', 'GATYATMAK MERU VAKRASANA', 'SIDEWAYS VIEWING', 'MAKARASANA', 'PADMASANA', 'VAJRASANA',
#           'ARDHA CHANDRASANA', 'YOGAMUDRASANA', 'BHUJANGASANA', 'SAITHALYASANA', 'BHU NAMANASANA', 'SARVANGASANA', 'NATARAJASANA', 'POORNA BHUJANGASANA',
#             'KOORMASANA', 'POORNA SHALABHASANA', 'POORNA DHANURASANA', 'BANDHA HASTA UTTHANASANA ', 'SHAVA UDARAKARSHANASANA ', 'CHAKKI CHALANASANA ', 
#             'KASHTHA TAKSHANASANA ', 'VAYU NISHKASANA', 'USHTRASANA', 'Samakonasana ', 'Matsyasana', 'Kandharasana', ' Setu Asana ', 'Paschimottanasana', 
#             'Meru Akarshanasana', 'Pada Hastasana', 'Seetkari Pranayama', 'Jalandhara Bandha', 'Tadagi Mudra', 'Maha Vedha Mudra', 'Shashankasana', 'JANU CHAKRA', 
#             'POORNA TITALI ASANA', 'MANIBANDHA CHAKRA', 'SKANDHA CHAKRA', 'GREEVA SANCHALANA', 'PADACHAKRASANA', 'PADA SANCHALANASANA', 'SUPTA PAWANMUKTASANA', 
#             'JHULANA LURHAKANASANA', 'SUPTA UDARAKARSHANASANA', 'NAUKASANA', 'RAJJU KARSHANASANA', 'GATYATMAK MERU VAKRASANA', 'NAUKA SANCHALANASANA', 'NAMASKARASANA', 
#             'KAUVA CHALASANA', 'PALMING', 'FRONT AND SIDEWAYS VIEWING', 'UP AND DOWN VIEWING', 'SHAVASANA', 'Advasana', 'ARDHA PADMASANA', 'SIDDHASANA', 'SIDDHA YONI ASANA', 
#             'SWASTIKASANA', 'PADADHIRASANA', 'VAJRASANA', 'ANANDA MADIRASANA', 'BHADRASANA', 'SIMHASANA', 'SIMHAGARJANASANA', 'VEERASANA', 'MARJARI-ASANA', 'VYAGHRASANA', 
#             'SHASHANK BHUJANGASANA', 'NAMAN PRANAMASANA', 'ASHWA SANCHALANASANA', 'ARDHA USHTRASANA', 'SUPTA VAJRASANA', 'AKARNA DHANURASANA', 'TADASANA', 'TIRYAK TADASANA', 
#             'KATI CHAKRASANA', 'TIRYAK KATI CHAKRASANA', 'MERU PRISHTHASANA', 'UTTHANASANA', 'DRUTA UTKATASANA', 'DWIKONASANA', 'TRIKONASANA', 'UTTHITA LOLASANA', 'DOLASANA', 
#             'ASHTANGA NAMASKARA', 'GUPTA PADMASANA', 'BADDHA PADMASANA', 'LOLASANA', 'KUKKUTASANA', 'GARBHASANA', 'TOLANGULASANA', 'SARAL BHUJANGASANA', 'TIRYAK BHUJANGASANA',
#               'SARPASANA', 'ARDHA SHALABHASANA', 'SHALABHASANA', 'SARAL DHANURASANA', 'DHANURASANA', 'ARDHA CHANDRASANA', 'UTTHAN PRISTHASANA', 'GOMUKHASANA', 'GATYATMAK PASCHIMOTTANASANA', 
#               'PADA PRASAR PASCHIMOTTANASANA', 'JANU SIRSHASANA', 'ARDHA PADMA PASCHIMOTTANASANA', 'HASTA PADA ANGUSHTHASANA', 'PADAHASTASANA', 'SIRSHA ANGUSTHA YOGASANA', 'UTTHITA JANU SIRSHASANA',
#                 'EKA PADOTTANASANA', 'MERU WAKRASANA', 'ARDHA MATSYENDRASANA', 'PARIVRITTI JANU SIRSHASANA', 'BHUMI PADA MASTAKASANA', 'MOORDHASANA', 'SARVANGASANA', 'PADMA SARVANGASANA', 'POORWA HALASANA', 
#                 'HALASANA', 'DRUTA HALASANA', 'ARDHA PADMA HALASANA', ' Stambhan Asana ', ' Sirshasana', ' SALAMBA SIRSHASANA', ' NIRALAMBA SIRSHASANA', 'OORDHWA PADMASANA', ' KAPALI ASANA', 
#                 ' EKA PADA PRANAMASANA', ' NATAVARASANA', ' GARUDASANA', ' TANDAVASANA', ' SARAL NATARAJASANA', ' NATARAJASANA', ' EKA PADASANA', ' BAKASANA', ' UTTHITA HASTA PADANGUSTHASANA', 
#                 ' MERUDANDASANA', ' NIRALAMBA PASCHIMOTTANASANA', ' ARDHA PADMA PADOTTANASANA', ' ARDHA BADDHA PADMOTTANASANA', ' VATAYANASANA', ' PADA ANGUSHTHASANA', ' BAKA DHYANASANA',
#                   ' Eka Pada Baka Dhyanasana', 'DWI HASTA BHUJASANA', 'EKA HASTA BHUJASANA', ' HAMSASANA', ' SANTOLANASANA', ' VASHISHTHASANA', 'POORNA BHUJANGASANA', ' KOORMASANA', ' POORNA SHALABHASANA', 
#                   ' POORNA DHANURASANA', ' DHANURAKARSHANASANA', ' PRISHTHASANA', ' PARIGHASANA', ' CHAKRASANA', ' HANUMANASANA', ' BRAHMACHARYASANA', ' GRIVASANA', ' SIRSHAPADA BHUMI SPARSHASANA', ' POORNA MATSYENDRASANA', 
#                   ' MAYURASANA', ' PADMA MAYURASANA', ' MOOLABANDHASANA', ' GORAKSHASANA', ' ASTAVAKRASANA', ' VRISCHIKASANA', ' EKA PADA SIRASANA', ' UTTHAN EKA PADA SIRASANA', ' DWI PADA SIRASANA', ' DWI PADA KANDHARASANA',
#                     ' PADMA PARVATASANA', 'VISHWAMITRASANA', 'Sukhanasana', 'Shavasana', 'Naukasana', 'Dhanurasana', 'Bhujangasana', 'Vakrasana', 'Bakasana', 'Halasana', 'Sarvangasana', 'Sirshasana', 'Gomukhasana', 'Upavistha Konasana', 
#                     'Utthan Pristhasana', 'Hasta Utthanasana', 'Paschimottanasana ', 'Kapalabhati pranayama', 'Padmasana', 'Taadasana', 'Ardha matsyendrasan', 'Balasana ', 'Phalakasana', 'Trikonasana', 'Setu Bandha ', 'Ustrasana', 'Natrajasana ',
#                       'Surya Namaskar', 'Bhairavasana', 'Hanumanasana', 'Vajrasana', 'Siddhasana', 'Virabhadrasana', 'Parvatasana', 'Camatkarasana', 'Vrikshasana', 'Tri Pada Adho Mukha Svanasana', 'Dandasana', 'Purvottanasana', 
#                       'bharmanasana', 'Ashtanga Namaskara', 'Pawanmuktasana', 'Ashta Chandrasana', 'Salabhasana', 'utkatasana', 'anjenayasana', 'Kraunchasana', 'Svastikasana', 'Sucirandhrasana', 'Supta Trivikramasana', 'Anantasna', 
#                       'Jathara Parivartanasana', 'Bananasana', 'Mayurasna', 'rajakapotasana', 'Bharadvajasana', 'Parighasana', 'Bhekasana', 'Kurmasana', 'Supta Padangusthasana A Straps', 'Ardha Kapotasana', 'Utkata Konasana', 'Sahaja Navasana', 
#                       'Skandasana', 'Parighasana', 'Anuvittasana', 'Jhulana Lurhakanasana', 'Bitilasana ', 'Karnapidasana', 'Lolasana', 'Sama Vritti', 'Prasarita Padottanasana', 'Yoganidrasana', 'Ardha Kapotasana', 'Kukkutasana', 'Marichyasana',
#                         'Virasana', 'Koundinyasana', 'Virabhadrasana', 'Mandukasana', 'Simhasana', 'Sitali', 'Murcha Pranayama', 'Hamsasana', 'Katichakrasana', 'Vayu Nishkasana', 'Gorakshasana', 'Pasasana ', 'kakimudra', 'kumbhakasana', 
#                         'makarasan', 'malasana', 'marjariasan', 'matsya Kridasana', 'padahastasana', 'padangusthasana', 'parivritta parsvakonasana', 'pincha mayurasana', 'pranamasana', 'shasankasana', 'supta udarakarshanasan', 'swastikasana', 
#                         'tolungulasana', 'utkata konasana', 'utthan prishthasana', 'vasisthasana', 'vipareet karni asana', 'vrishchikasana', 'mudrasana', 'ashta chandrasana', 'bhujapidasana', 'garudasana', 'kapotasan', 'muktasana', 'parshvakonasana',
#                           'prasarita padottasana', 'samakonasana', 'tittibhasana', 'supta trivikramasana', 'tulasana', 'Upavistha Titli Asana', 'uttana shishoasana', 'agnistambhasana', 'mrigi mudra', 'abhaya hridiya mudra', 'Viparita Shalabhasana', 
#                           'gandha bherudasana', 'karandavasana', 'moordhasana', 'dwi pada sirsasana', 'swarga dwijasana', 'ashwathasana', 'Yastikasana', 'nataprathanasana', 'Chakki Chalanasana', 'ushtrasana', 'talasana', 'padma mudra', 'brahmandasana', 
#                           'Jala Neti ( Shatkarma )', 'sheetkrama kapalbhati', 'vastra dhauti', 'trataka', 'somachandrasana', 'Gherandasana ', 'kapilasana', 'omkarasana', 'kashyapawsana', 'bhunamanasana', 'Mandalasana']

# for i in poses:
#   pi.download(i,limit=5)  
# in cmd to make all the images from x format to y format (*. *x) 
import os 
def move_files(sourceFolder,targetFolder):
    try:
        for path, dir , files in os.walk(sourceFolder):
            if files:
                for file in files:
                    if not os.path.isfile(targetFolder+file):
                        os.rename(path+'\\'+file, targetFolder+file)
        print("all files moved")
    except Exception as e :
        print(e)
 
sourceFolder = r'D:\VIT[college work]\4-2mainproject\yogaposturedetection\static'
targetFolder = r'D:\VIT[college work]\4-2mainproject\yogaposturedetection\static\images'
# move_files(sourceFolder,targetFolder)
# for path,dir,files in os.walk(sourceFolder):
#     print(dir)
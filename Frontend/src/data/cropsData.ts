
import { CropData } from '../types/crop';

export const cropsData: CropData[] = [
  {
    id: '1',
    name: {
      en: 'Rice',
      mr: 'तांदूळ'
    },
    fertilizers: [
      {
        id: 'f1',
        name: {
          en: 'NPK 10-26-26',
          mr: 'एनपीके १०-२६-२६'
        },
        image: '/images/images/NPK.jpg',
        type: {
          en: 'Complex Fertilizer',
          mr: 'मिश्र खत'
        },
        usage: {
          en: 'Provides balanced nutrients for rice during grain formation.',
          mr: 'धान्य निर्मितीच्या दरम्यान तांदळासाठी संतुलित पोषक तत्वे पुरवते.'
        },
        benefits: {
          en: [
            'Enhances root development',
            'Improves grain filling',
            'Increases yield potential'
          ],
          mr: [
            'मुळांचा विकास वाढवते',
            'दाणे भरण्यात सुधारणा करते',
            'उत्पादन क्षमता वाढवते'
          ]
        },
        application: {
          en: 'Apply 100-125 kg/hectare during field preparation or transplanting.',
          mr: 'शेत तयारी किंवा रोपे लावताना प्रति हेक्टर १००-१२५ किलो वापरा.'
        },
        price: '₹1,200 - ₹1,500',
        link: 'https://www.amazon.in/NPK-Fertilizer/s?k=NPK+Fertilizer'
      },
      {
        id: 'f2',
        name: {
          en: 'Urea',
          mr: 'युरिया'
        },
        image: '/images/images/Urea.jpg',
        type: {
          en: 'Nitrogen Fertilizer',
          mr: 'नायट्रोजन खत'
        },
        usage: {
          en: 'Primary source of nitrogen for plant growth and development.',
          mr: 'वनस्पतीच्या वाढीसाठी आणि विकासासाठी नायट्रोजनचा प्राथमिक स्रोत.'
        },
        benefits: {
          en: [
            'Promotes vegetative growth',
            'Increases chlorophyll content',
            'Enhances protein synthesis'
          ],
          mr: [
            'वनस्पतीची वाढ प्रोत्साहित करते',
            'हरितद्रव्याचे प्रमाण वाढवते',
            'प्रथिन संश्लेषण वाढवते'
          ]
        },
        application: {
          en: 'Split application: 1/3 at transplanting, 1/3 at tillering, 1/3 at panicle initiation.',
          mr: 'विभाजित अनुप्रयोग: १/३ रोपे लावताना, १/३ फुटवे येताना, १/३ कणीस निर्मिती वेळी.'
        },
        price: '₹300 - ₹400',
        link: 'https://www.amazon.in/Urea-Fertilizer/s?k=Urea+Fertilizer'
      }
    ],
    pesticides: [
      {
        id: 'p1',
        name: {
          en: 'Carbofuran 3G',
          mr: 'कार्बोफ्युरान ३जी'
        },
        image: '/images/images/Carbofuran.jpeg',
        type: {
          en: 'Systemic Insecticide',
          mr: 'सिस्टेमिक कीटकनाशक'
        },
        usage: {
          en: 'Controls stem borers, leaf folders, and root pests in rice.',
          mr: 'तांदळामध्ये खोड कीड, पान वाळणारी कीड आणि मुळांवरील कीड नियंत्रित करते.'
        },
        benefits: {
          en: [
            'Long-lasting protection',
            'Controls multiple pests',
            'Easy to apply'
          ],
          mr: [
            'दीर्घकालीन संरक्षण',
            'अनेक कीटक नियंत्रित करते',
            'वापरण्यास सोपे'
          ]
        },
        application: {
          en: 'Apply 10-12 kg/hectare in standing water in rice fields.',
          mr: 'भात शेतात उभ्या पाण्यात प्रति हेक्टर १०-१२ किलो वापरा.'
        },
        price: '₹200 - ₹250',
        link: 'https://www.amazon.in/Insecticide-Granules/s?k=Insecticide+Granules'
      },
      {
        id: 'p2',
        name: {
          en: 'Tricyclazole 75% WP',
          mr: 'ट्रायसायक्लाझोल ७५% डब्ल्यूपी'
        },
        image: '/images/images/Tricyclazole.jpeg',
        type: {
          en: 'Fungicide',
          mr: 'बुरशीनाशक'
        },
        usage: {
          en: 'Controls blast disease in rice.',
          mr: 'तांदळावरील ब्लास्ट रोग नियंत्रित करते.'
        },
        benefits: {
          en: [
            'Preventive and curative action',
            'Protects new growth',
            'Improves grain quality'
          ],
          mr: [
            'प्रतिबंधात्मक आणि उपचारात्मक कृती',
            'नवीन वाढ संरक्षित करते',
            'धान्याची गुणवत्ता सुधारते'
          ]
        },
        application: {
          en: 'Apply 0.6 g/liter of water at first sign of disease, repeat after 15 days if needed.',
          mr: 'रोगाच्या पहिल्या लक्षणांवर प्रति लीटर पाण्यात ०.६ ग्रॅम वापरा, आवश्यक असल्यास १५ दिवसांनंतर पुन्हा वापरा.'
        },
        price: '₹450 - ₹500',
        link: 'https://www.amazon.in/Fungicides/s?k=Fungicides+for+Rice'
      }
    ]
  },
  {
    id: '2',
    name: {
      en: 'Wheat',
      mr: 'गहू'
    },
    fertilizers: [
      {
        id: 'f3',
        name: {
          en: 'DAP (Di-Ammonium Phosphate)',
          mr: 'डीएपी (डाय-अमोनियम फॉस्फेट)'
        },
        image: '/images/images/dap.jpg',
        type: {
          en: 'Phosphatic Fertilizer',
          mr: 'फॉस्फेटिक खत'
        },
        usage: {
          en: 'Provides phosphorus and nitrogen for root development and early growth.',
          mr: 'मुळांच्या विकासासाठी आणि प्रारंभिक वाढीसाठी फॉस्फरस आणि नायट्रोजन पुरवते.'
        },
        benefits: {
          en: [
            'Promotes root growth',
            'Enhances tillering',
            'Improves drought resistance'
          ],
          mr: [
            'मुळांची वाढ प्रोत्साहित करते',
            'फुटवे येण्यास मदत करते',
            'दुष्काळ प्रतिकारशक्ती वाढवते'
          ]
        },
        application: {
          en: 'Apply 100-125 kg/hectare as basal dose before sowing.',
          mr: 'पेरणीपूर्वी पायाभूत मात्रा म्हणून प्रति हेक्टर १००-१२५ किलो वापरा.'
        },
        price: '₹1,200 - ₹1,400',
        link: 'https://www.amazon.in/DAP-Fertilizer/s?k=DAP+Fertilizer'
      }
    ],
    pesticides: [
      {
        id: 'p3',
        name: {
          en: 'Propiconazole 25% EC',
          mr: 'प्रोपिकोनाझोल २५% ईसी'
        },
        image: '/images/images/Propiconazole.jpg',
        type: {
          en: 'Fungicide',
          mr: 'बुरशीनाशक'
        },
        usage: {
          en: 'Controls rust, powdery mildew, and leaf spot diseases in wheat.',
          mr: 'गव्हामधील तांबेरा, भुरी आणि पानावरील ठिपके रोग नियंत्रित करते.'
        },
        benefits: {
          en: [
            'Systemic action',
            'Protective and curative',
            'Long-lasting effect'
          ],
          mr: [
            'सिस्टेमिक क्रिया',
            'संरक्षणात्मक आणि उपचारात्मक',
            'दीर्घकालीन परिणाम'
          ]
        },
        application: {
          en: 'Apply 1 ml/liter of water at first sign of disease. Repeat after 15-20 days if needed.',
          mr: 'रोगाच्या पहिल्या लक्षणांवर प्रति लीटर पाण्यात १ मिली वापरा. आवश्यक असल्यास १५-२० दिवसांनंतर पुन्हा वापरा.'
        },
        price: '₹550 - ₹600',
        link: 'https://www.amazon.in/Fungicides/s?k=Fungicides+for+Wheat'
      }
    ]
  },
  {
    id: '3',
    name: {
      en: 'Cotton',
      mr: 'कापूस'
    },
    fertilizers: [
      {
        id: 'f4',
        name: {
          en: 'NPK 20-20-20',
          mr: 'एनपीके २०-२०-२०'
        },
        image: '/images/images/npk20-20-20.jpeg',
        type: {
          en: 'Water Soluble Fertilizer',
          mr: 'पाण्यात विरघळणारे खत'
        },
        usage: {
          en: 'Balanced nutrition for cotton throughout the growing season.',
          mr: 'वाढीच्या संपूर्ण हंगामात कापसासाठी संतुलित पोषण.'
        },
        benefits: {
          en: [
            'Promotes balanced growth',
            'Enhances boll development',
            'Improves fiber quality'
          ],
          mr: [
            'संतुलित वाढ प्रोत्साहित करते',
            'कापसाच्या बोंडाचा विकास वाढवते',
            'धाग्याची गुणवत्ता सुधारते'
          ]
        },
        application: {
          en: 'Apply 5-10 g/liter of water through foliar spray at square formation and boll development stages.',
          mr: 'चौकोन निर्मिती आणि बोंड विकास अवस्थेत पानांवर फवारणी द्वारे प्रति लीटर पाण्यात ५-१० ग्रॅम वापरा.'
        },
        price: '₹300 - ₹400',
        link: 'https://www.amazon.in/NPK-Fertilizer/s?k=NPK+20+20+20+Fertilizer'
      }
    ],
    pesticides: [
      {
        id: 'p4',
        name: {
          en: 'Imidacloprid 17.8% SL',
          mr: 'इमिडाक्लोप्रिड १७.८% एसएल'
        },
        image: '/images/images/Imidacloprid.jpeg',
        type: {
          en: 'Systemic Insecticide',
          mr: 'सिस्टेमिक कीटकनाशक'
        },
        usage: {
          en: 'Controls sucking pests like aphids, jassids, and whiteflies in cotton.',
          mr: 'कापसावरील मावा, तुडतुडे आणि पांढरी माशी सारखे चोखणारे कीड नियंत्रित करते.'
        },
        benefits: {
          en: [
            'Rapid action',
            'Long residual control',
            'Low application rate'
          ],
          mr: [
            'जलद क्रिया',
            'दीर्घकालीन नियंत्रण',
            'कमी वापर दर'
          ]
        },
        application: {
          en: 'Apply 0.5 ml/liter of water when pest population reaches economic threshold.',
          mr: 'कीटक संख्या आर्थिक पातळीवर पोहोचल्यावर प्रति लीटर पाण्यात ०.५ मिली वापरा.'
        },
        price: '₹400 - ₹500',
        link: 'https://www.amazon.in/Insecticides/s?k=Imidacloprid+Insecticide'
      }
    ]
  }
];

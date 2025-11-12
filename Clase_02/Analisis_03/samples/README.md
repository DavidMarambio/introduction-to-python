About Dataset
Context

Water is essential for all the crops and livestocks as much as it is important for human beings. Crops and livestocks consume direct ground water, and if the quality is not up to the mark, or becomes unusable, then crops and livestocks are affected, which may lead to crop failure or livestocks developing diseases.
By knowing the quality of the ground water, whether usable or not, the water can be put to appropriate use. Farmers can grow specific crops which can tolerate that quality of water.

Usage

By using these datasets one can train an ML Classification model to classify the water quality into one of the multiple classes and know where and how the water can be used.

Data collected

This data is collected from Telangana Open Data portal, Telangana State, India.
This data contains samples tested from various districts.

There are 3 files, each or year 2018, 2019 and 2020 contains post-monsoon season groundwater quality details.
ground_water_quality_2018_post.csv,
ground_water_quality_2019_post.csv,
ground_water_quality_2020_post.csv can be combinedly used.

Each dataset contains 26 columns such as:
serial num (sno), District, Mandal, Village, Lattitude, Longitude, Chemicals (such as Ca, Mg, CO3 etc), Total Hardness of the water, Total dissolved solids, RSC, SAR, and the target variables 'Classification' and 'Classification1'.
The feature columns can be used to predict water quality which are broadly classified into 9 classes, which are: C3S1, C2S1, C4S1,C4S2 , C3S2 ,C4S4 ,C3S3 ,C4S3, C1S1.

C1S1: Low salinity and low sodium waters are good for irrigation and can be used with
most crops with no restriction on use on most of the soils.

C2S1: Medium salinity and low sodium waters are good for irrigation and can be used on
all most all soils with little danger of development of harmful levels of exchangeable
sodium if a moderate amount of leaching occurs. Crops can be grown without any special
consideration for salinity control.

C3S1: The high salinity and low sodium waters require good drainage. Crops with good
salt tolerance should be selected.

C3S2: The high salinity and medium sodium waters require good drainage and can be used
on coarse - textured or organic soils having good permeability.

C3S3: These high salinity and high sodium waters require special soil management, good
drainage, high leaching and organic matter additions. Gypsum amendments make feasible
the use of these waters.

C4S1: Very high salinity and low sodium waters are not suitable for irrigation unless the
soil must be permeable and drainage must be adequate. Irrigation waters must be applied
in excess to provide considerable leaching. Salt tolerant crops must be selected.

C4S2: Very high salinity and medium sodium waters are not suitable for irrigation on fine
textured soils and low leaching conditions and can be used for irrigation on coarse textured
or organic soils having good permeability.

C4S3: Very high salinity and high sodium waters produce harmful levels of exchangeable
sodium in most soils and will require special soil management, good drainage, high
leaching, and organic matter additions. The Gypsum amendment makes feasible the use of
these waters.

C4S4: Very high salinity and very high sodium waters are generally unsuitable for
irrigation purposes. These are sodium chloride types of water and can cause sodium
hazards. It can be used on coarse-textured soils with very good drainage for very high salt tolerant crops. Gypsum amendments make feasible the use of these waters.

Classification of groundwater based on RSC

RSC is defined as the excess of carbonate and bicarbonate amount over the alkaline
earths (Ca2+ and Mg2+). Use of RSC beyond permissible limit (>2.5) adversely affects irrigation.
The tendency of Ca2+ and Mg2+ to precipitate, as the water in the soil becomes more
concentrated, as a result of evaporation and plant transpiration, and gets fixed in the soil by the
process of base exchange, thereby decreasing the soil permeability.
RSC = ((CO3 2-) + (HCO3-)) - ((Ca2+)+(Mg2+))
Where concentrations are in meq/L.

RSC less than 1.25 is safe
RSC between 1.25 and 2.50 is marginal
RSC greater than 2.50 is unsuitable

Use of ground water for livestock and poultry

TDS < 1000 mg/L -- Excellent -- Excellent for all classes of livestock and poultry.

TDS b/w 1000-3000 -- very satisfactory -- Satisfactory for all classes of livestock. May cause temporary mild diarrhea in livestock not accustomed to them. Those waters approaching the upper limits may cause some watery droppings in poultry.

TDS b/w 3000-5000 -- Satisfactory for livestock Unfit for poultry -- Satisfactory for livestock but may be refused by animals not accustomed to it. If Sulphate salts predominate, animals may show temporary diarrhea. Poor waters for poultry, often causing watery faeces, increased mortality and decreasedgrowth especially in turkeys.

TDS b/w 5000-7000 -- Limited use for livestock Unfit for poultry -- This water can be used for livestock except for those that are pregnant or lactating. It may have some laxative effect and may be refused by animals until they become accustomed to it. It is unsatisfactory for poultry.

TDS b/w 7000-10,000 -- Very limited use -- Considerable risk for pregnant and lactating cows, horses, sheep and for the young of these species. It may be used for older ruminants or horses. Unfit for poultry and probably swine.

TDS > 10,000 -- Not recommended -- This water is unsatisfactory for all classes of livestock and poultry.
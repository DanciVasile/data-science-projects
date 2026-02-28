# 📖 Ames Housing Dataset — Data Dictionary

> Complete reference for all **82 features** in the Ames Housing dataset, including value codes and descriptions.

---

## 🏗️ Building & Dwelling

### MSSubClass
Identifies the type of dwelling involved in the sale.

| Code | Description |
|------|-------------|
| 20 | 1-Story 1946 & Newer All Styles |
| 30 | 1-Story 1945 & Older |
| 40 | 1-Story W/Finished Attic All Ages |
| 45 | 1-1/2 Story — Unfinished All Ages |
| 50 | 1-1/2 Story Finished All Ages |
| 60 | 2-Story 1946 & Newer |
| 70 | 2-Story 1945 & Older |
| 75 | 2-1/2 Story All Ages |
| 80 | Split or Multi-Level |
| 85 | Split Foyer |
| 90 | Duplex — All Styles and Ages |
| 120 | 1-Story PUD (Planned Unit Development) — 1946 & Newer |
| 150 | 1-1/2 Story PUD — All Ages |
| 160 | 2-Story PUD — 1946 & Newer |
| 180 | PUD — Multilevel — Incl Split Lev/Foyer |
| 190 | 2 Family Conversion — All Styles and Ages |

### BldgType
Type of dwelling.

| Code | Description |
|------|-------------|
| 1Fam | Single-family Detached |
| 2FmCon | Two-family Conversion; originally built as one-family dwelling |
| Duplx | Duplex |
| TwnhsE | Townhouse End Unit |
| TwnhsI | Townhouse Inside Unit |

### HouseStyle
Style of dwelling.

| Code | Description |
|------|-------------|
| 1Story | One story |
| 1.5Fin | One and one-half story: 2nd level finished |
| 1.5Unf | One and one-half story: 2nd level unfinished |
| 2Story | Two story |
| 2.5Fin | Two and one-half story: 2nd level finished |
| 2.5Unf | Two and one-half story: 2nd level unfinished |
| SFoyer | Split Foyer |
| SLvl | Split Level |

---

## 📍 Location & Zoning

### MSZoning
General zoning classification of the sale.

| Code | Description |
|------|-------------|
| A | Agriculture |
| C | Commercial |
| FV | Floating Village Residential |
| I | Industrial |
| RH | Residential High Density |
| RL | Residential Low Density |
| RP | Residential Low Density Park |
| RM | Residential Medium Density |

### Neighborhood
Physical locations within Ames city limits.

| Code | Description |
|------|-------------|
| Blmngtn | Bloomington Heights |
| Blueste | Bluestem |
| BrDale | Briardale |
| BrkSide | Brookside |
| ClearCr | Clear Creek |
| CollgCr | College Creek |
| Crawfor | Crawford |
| Edwards | Edwards |
| Gilbert | Gilbert |
| IDOTRR | Iowa DOT and Rail Road |
| MeadowV | Meadow Village |
| Mitchel | Mitchell |
| Names | North Ames |
| NoRidge | Northridge |
| NPkVill | Northpark Villa |
| NridgHt | Northridge Heights |
| NWAmes | Northwest Ames |
| OldTown | Old Town |
| SWISU | South & West of Iowa State University |
| Sawyer | Sawyer |
| SawyerW | Sawyer West |
| Somerst | Somerset |
| StoneBr | Stone Brook |
| Timber | Timberland |
| Veenker | Veenker |

### Condition1
Proximity to various conditions.

| Code | Description |
|------|-------------|
| Artery | Adjacent to arterial street |
| Feedr | Adjacent to feeder street |
| Norm | Normal |
| RRNn | Within 200' of North-South Railroad |
| RRAn | Adjacent to North-South Railroad |
| PosN | Near positive off-site feature (park, greenbelt, etc.) |
| PosA | Adjacent to positive off-site feature |
| RRNe | Within 200' of East-West Railroad |
| RRAe | Adjacent to East-West Railroad |

### Condition2
Proximity to various conditions (if more than one is present). Same codes as Condition1.

---

## 📐 Lot Features

### LotFrontage
Linear feet of street connected to property. *(numeric)*

### LotArea
Lot size in square feet. *(numeric)*

### Street
Type of road access to property.

| Code | Description |
|------|-------------|
| Grvl | Gravel |
| Pave | Paved |

### Alley
Type of alley access to property.

| Code | Description |
|------|-------------|
| Grvl | Gravel |
| Pave | Paved |
| NA | No alley access |

### LotShape
General shape of property.

| Code | Description |
|------|-------------|
| Reg | Regular |
| IR1 | Slightly irregular |
| IR2 | Moderately Irregular |
| IR3 | Irregular |

### LandContour
Flatness of the property.

| Code | Description |
|------|-------------|
| Lvl | Near Flat/Level |
| Bnk | Banked — Quick and significant rise from street grade to building |
| HLS | Hillside — Significant slope from side to side |
| Low | Depression |

### LotConfig
Lot configuration.

| Code | Description |
|------|-------------|
| Inside | Inside lot |
| Corner | Corner lot |
| CulDSac | Cul-de-sac |
| FR2 | Frontage on 2 sides of property |
| FR3 | Frontage on 3 sides of property |

### LandSlope
Slope of property.

| Code | Description |
|------|-------------|
| Gtl | Gentle slope |
| Mod | Moderate slope |
| Sev | Severe slope |

### Utilities
Type of utilities available.

| Code | Description |
|------|-------------|
| AllPub | All public Utilities (E, G, W, & S) |
| NoSewr | Electricity, Gas, and Water (Septic Tank) |
| NoSeWa | Electricity and Gas Only |
| ELO | Electricity only |

---

## 🏠 Quality & Condition Ratings

### OverallQual
Rates the overall material and finish of the house.

| Rating | Description |
|--------|-------------|
| 10 | Very Excellent |
| 9 | Excellent |
| 8 | Very Good |
| 7 | Good |
| 6 | Above Average |
| 5 | Average |
| 4 | Below Average |
| 3 | Fair |
| 2 | Poor |
| 1 | Very Poor |

### OverallCond
Rates the overall condition of the house. Same scale as OverallQual.

### Functional
Home functionality (assume typical unless deductions are warranted).

| Code | Description |
|------|-------------|
| Typ | Typical Functionality |
| Min1 | Minor Deductions 1 |
| Min2 | Minor Deductions 2 |
| Mod | Moderate Deductions |
| Maj1 | Major Deductions 1 |
| Maj2 | Major Deductions 2 |
| Sev | Severely Damaged |
| Sal | Salvage only |

---

## 📅 Year & Date

| Feature | Description |
|---------|-------------|
| **YearBuilt** | Original construction date |
| **YearRemodAdd** | Remodel date (same as construction date if no remodeling) |
| **MoSold** | Month sold (MM) |
| **YrSold** | Year sold (YYYY) |

---

## 🏗️ Roof

### RoofStyle
Type of roof.

| Code | Description |
|------|-------------|
| Flat | Flat |
| Gable | Gable |
| Gambrel | Gambrel (Barn) |
| Hip | Hip |
| Mansard | Mansard |
| Shed | Shed |

### RoofMatl
Roof material.

| Code | Description |
|------|-------------|
| ClyTile | Clay or Tile |
| CompShg | Standard (Composite) Shingle |
| Membran | Membrane |
| Metal | Metal |
| Roll | Roll |
| Tar&Grv | Gravel & Tar |
| WdShake | Wood Shakes |
| WdShngl | Wood Shingles |

---

## 🧱 Exterior

### Exterior1st / Exterior2nd
Exterior covering on house (1st and 2nd material).

| Code | Description |
|------|-------------|
| AsbShng | Asbestos Shingles |
| AsphShn | Asphalt Shingles |
| BrkComm | Brick Common |
| BrkFace | Brick Face |
| CBlock | Cinder Block |
| CemntBd | Cement Board |
| HdBoard | Hard Board |
| ImStucc | Imitation Stucco |
| MetalSd | Metal Siding |
| Other | Other |
| Plywood | Plywood |
| PreCast | PreCast |
| Stone | Stone |
| Stucco | Stucco |
| VinylSd | Vinyl Siding |
| Wd Sdng | Wood Siding |
| WdShing | Wood Shingles |

### MasVnrType
Masonry veneer type.

| Code | Description |
|------|-------------|
| BrkCmn | Brick Common |
| BrkFace | Brick Face |
| CBlock | Cinder Block |
| None | None |
| Stone | Stone |

### MasVnrArea
Masonry veneer area in square feet. *(numeric)*

### ExterQual
Quality of the material on the exterior.

| Code | Description |
|------|-------------|
| Ex | Excellent |
| Gd | Good |
| TA | Average/Typical |
| Fa | Fair |
| Po | Poor |

### ExterCond
Present condition of the material on the exterior. Same scale as ExterQual.

---

## 🏚️ Basement

### Foundation
Type of foundation.

| Code | Description |
|------|-------------|
| BrkTil | Brick & Tile |
| CBlock | Cinder Block |
| PConc | Poured Concrete |
| Slab | Slab |
| Stone | Stone |
| Wood | Wood |

### BsmtQual
Height of the basement.

| Code | Description |
|------|-------------|
| Ex | Excellent (100+ inches) |
| Gd | Good (90–99 inches) |
| TA | Typical (80–89 inches) |
| Fa | Fair (70–79 inches) |
| Po | Poor (<70 inches) |
| NA | No Basement |

### BsmtCond
General condition of the basement.

| Code | Description |
|------|-------------|
| Ex | Excellent |
| Gd | Good |
| TA | Typical — slight dampness allowed |
| Fa | Fair — dampness or some cracking or settling |
| Po | Poor — Severe cracking, settling, or wetness |
| NA | No Basement |

### BsmtExposure
Walkout or garden level walls.

| Code | Description |
|------|-------------|
| Gd | Good Exposure |
| Av | Average Exposure |
| Mn | Minimum Exposure |
| No | No Exposure |
| NA | No Basement |

### BsmtFinType1 / BsmtFinType2
Rating of basement finished area.

| Code | Description |
|------|-------------|
| GLQ | Good Living Quarters |
| ALQ | Average Living Quarters |
| BLQ | Below Average Living Quarters |
| Rec | Average Rec Room |
| LwQ | Low Quality |
| Unf | Unfinished |
| NA | No Basement |

### Basement Numeric Features

| Feature | Description |
|---------|-------------|
| **BsmtFinSF1** | Type 1 finished square feet |
| **BsmtFinSF2** | Type 2 finished square feet |
| **BsmtUnfSF** | Unfinished square feet of basement area |
| **TotalBsmtSF** | Total square feet of basement area |
| **BsmtFullBath** | Basement full bathrooms |
| **BsmtHalfBath** | Basement half bathrooms |

---

## 🔥 Heating & Electrical

### Heating
Type of heating.

| Code | Description |
|------|-------------|
| Floor | Floor Furnace |
| GasA | Gas forced warm air furnace |
| GasW | Gas hot water or steam heat |
| Grav | Gravity furnace |
| OthW | Hot water or steam heat other than gas |
| Wall | Wall furnace |

### HeatingQC
Heating quality and condition.

| Code | Description |
|------|-------------|
| Ex | Excellent |
| Gd | Good |
| TA | Average/Typical |
| Fa | Fair |
| Po | Poor |

### CentralAir
Central air conditioning.

| Code | Description |
|------|-------------|
| N | No |
| Y | Yes |

### Electrical
Electrical system.

| Code | Description |
|------|-------------|
| SBrkr | Standard Circuit Breakers & Romex |
| FuseA | Fuse Box over 60 AMP and all Romex wiring (Average) |
| FuseF | 60 AMP Fuse Box and mostly Romex wiring (Fair) |
| FuseP | 60 AMP Fuse Box and mostly knob & tube wiring (Poor) |
| Mix | Mixed |

---

## 📏 Living Area & Rooms

| Feature | Description |
|---------|-------------|
| **1stFlrSF** | First floor square feet |
| **2ndFlrSF** | Second floor square feet |
| **LowQualFinSF** | Low quality finished square feet (all floors) |
| **GrLivArea** | Above grade (ground) living area square feet |
| **FullBath** | Full bathrooms above grade |
| **HalfBath** | Half baths above grade |
| **Bedroom** | Bedrooms above grade (does NOT include basement bedrooms) |
| **Kitchen** | Kitchens above grade |
| **TotRmsAbvGrd** | Total rooms above grade (does not include bathrooms) |

### KitchenQual
Kitchen quality.

| Code | Description |
|------|-------------|
| Ex | Excellent |
| Gd | Good |
| TA | Typical/Average |
| Fa | Fair |
| Po | Poor |

---

## 🔥 Fireplace

### Fireplaces
Number of fireplaces. *(numeric)*

### FireplaceQu
Fireplace quality.

| Code | Description |
|------|-------------|
| Ex | Excellent — Exceptional Masonry Fireplace |
| Gd | Good — Masonry Fireplace in main level |
| TA | Average — Prefabricated Fireplace in main living area or Masonry in basement |
| Fa | Fair — Prefabricated Fireplace in basement |
| Po | Poor — Ben Franklin Stove |
| NA | No Fireplace |

---

## 🚗 Garage

### GarageType
Garage location.

| Code | Description |
|------|-------------|
| 2Types | More than one type of garage |
| Attchd | Attached to home |
| Basment | Basement Garage |
| BuiltIn | Built-In (typically has room above garage) |
| CarPort | Car Port |
| Detchd | Detached from home |
| NA | No Garage |

### GarageYrBlt
Year garage was built. *(numeric)*

### GarageFinish
Interior finish of the garage.

| Code | Description |
|------|-------------|
| Fin | Finished |
| RFn | Rough Finished |
| Unf | Unfinished |
| NA | No Garage |

### GarageCars
Size of garage in car capacity. *(numeric)*

### GarageArea
Size of garage in square feet. *(numeric)*

### GarageQual / GarageCond
Garage quality and condition.

| Code | Description |
|------|-------------|
| Ex | Excellent |
| Gd | Good |
| TA | Typical/Average |
| Fa | Fair |
| Po | Poor |
| NA | No Garage |

### PavedDrive
Paved driveway.

| Code | Description |
|------|-------------|
| Y | Paved |
| P | Partial Pavement |
| N | Dirt/Gravel |

---

## 🌳 Outdoor & Extras

| Feature | Description |
|---------|-------------|
| **WoodDeckSF** | Wood deck area in square feet |
| **OpenPorchSF** | Open porch area in square feet |
| **EnclosedPorch** | Enclosed porch area in square feet |
| **3SsnPorch** | Three season porch area in square feet |
| **ScreenPorch** | Screen porch area in square feet |
| **PoolArea** | Pool area in square feet |

### PoolQC
Pool quality.

| Code | Description |
|------|-------------|
| Ex | Excellent |
| Gd | Good |
| TA | Average/Typical |
| Fa | Fair |
| NA | No Pool |

### Fence
Fence quality.

| Code | Description |
|------|-------------|
| GdPrv | Good Privacy |
| MnPrv | Minimum Privacy |
| GdWo | Good Wood |
| MnWw | Minimum Wood/Wire |
| NA | No Fence |

### MiscFeature
Miscellaneous feature not covered in other categories.

| Code | Description |
|------|-------------|
| Elev | Elevator |
| Gar2 | 2nd Garage (if not described in garage section) |
| Othr | Other |
| Shed | Shed (over 100 SF) |
| TenC | Tennis Court |
| NA | None |

### MiscVal
Dollar value of miscellaneous feature. *(numeric)*

---

## 💰 Sale Information

### SaleType
Type of sale.

| Code | Description |
|------|-------------|
| WD | Warranty Deed — Conventional |
| CWD | Warranty Deed — Cash |
| VWD | Warranty Deed — VA Loan |
| New | Home just constructed and sold |
| COD | Court Officer Deed/Estate |
| Con | Contract 15% Down payment regular terms |
| ConLw | Contract Low Down payment and low interest |
| ConLI | Contract Low Interest |
| ConLD | Contract Low Down |
| Oth | Other |

### SaleCondition
Condition of sale.

| Code | Description |
|------|-------------|
| Normal | Normal Sale |
| Abnorml | Abnormal Sale — trade, foreclosure, short sale |
| AdjLand | Adjoining Land Purchase |
| Alloca | Allocation — two linked properties with separate deeds |
| Family | Sale between family members |
| Partial | Home was not completed when last assessed (associated with New Homes) |

### 🎯 SalePrice
**Target variable** — Sale price of the property in dollars. *(numeric)*

---

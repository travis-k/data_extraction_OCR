# Extracting Data from the Merlin Parts Catalog using Tesseract OCR and Point Set Registration
## Overview
The purpose of this project is to demonstrate one method of extracting data from old, scanned documents containing formatted information. The document used for this demonstration is the parts catalog for the Rolls Royce/Packard V-1650 Merlin aircraft engines, AN 02A-55AC-4. High-quality scans of this document are available through [AirCorps Library](https://aircorpslibrary.com/). Low-quality scans are readily found on the net.

This project uses the page format to segment the page into pre-defined areas, and extract this information. This is an overview:
1. The image is read in, and scrubbed to reveal the necessary features
2. Point set registration is used to match this scrubbed scan image to a premade template, giving us the necessary coordinate transformation (scale, rotation and translation) to move from our template coordinates to the scan coordinates, and vice versa
3. Using this transformation, different areas of the page are segmented and read in via OCR
4. The defining feature, the ''Part Number'' column, is used to break the page vertically, seperating the information for each part


## Walkthrough
A typical page from this document:

<img src="https://github.com/travis-k/data_extraction_OCR/blob/master/images/page_23.jpg" height="500">

Based on this typical page, I created a template which we will use with point set registration to identify the orientation of the scan, and locate the unique areas we need to extract information from.

<img src="https://github.com/travis-k/data_extraction_OCR/blob/master/images/template_tight.png" height="500">

The scanned-in page is too noisy as-is, so we scrub almost everything off of the page to reveal the underlying template. This ''cleaned page'' is only used in the point set registration, so the loss of information is not important. Part of the cleaning is done with code taken from noteshrink (https://github.com/mzucker/noteshrink). The rest is simply based on feature sizes, small features such as spots or letters are mostly removed.

<img src="https://github.com/travis-k/data_extraction_OCR/blob/master/images/cleaned_page.png" height="500">

Point set registration, using rigid_registration from pycpd (https://github.com/siavashk/pycpd) is then used to match the cleaned page to the template. This gives us the scale, rotation and translation needed to move from the scanned page coordinates to the template coordinates (in which we know where all the data should be) and vice versa. Below is the scanned page after being adjusted and cropped to fit the template format.

<img src="https://github.com/travis-k/data_extraction_OCR/blob/master/images/adj_page.png" height="500">

Now that we know the transformation, and where the data we want is on the template, we can extract different areas. First, we extract the main feature - the list of part numbers. The vertical spacing of these part numbers will be used to split the rest of the data up, so that we can extract the corresponding figure and index numbers, part descriptions, etc. 

<img src="https://github.com/travis-k/data_extraction_OCR/blob/master/images/img_part_number.png" height="500">

Using pytesseract, we can get the part numbers (as well as their locations on the scanned page)
```
['618064', '618001', '618046', '616457', '616873', '608758', '608782', '608760', '*608759', '608547', '608783', '608761', '608547', '608787', '*608763', '600191', '604674', '601091', '601443', '606345', '603589', '608742', '616383', '*600365', '600161', '600279', '616377', '615836', '600287', '600289', '600255']
```

From this point, we move section-by-section to extract text, knowing the location of what we want, and assign it to the part number that is on that vertical location. For this example, I also extracted the part description and the figure and index number, but this technique will also be used to log the units per assembly for each engine variant.

Below is the full output of the main.py script. There are imperfections, such as unhandled line breaks (such as 'super- sedes'). Some words are also missing, as some part descriptions are difficult to read due to the vertical lines on the left side of descriptions. 

Future work will invovle trying to increase the accuracy of the OCR and page segmentation, as well as including the parts count for the engine variants. 

```
Part Number: 618064
	Group: GROUP— Cylinder
	Description: Assembly - Complete cylinder block and head "A" unit with valves, valve cover, camshaft and rocker mechanism, part of coolant inlet section and part of camshaft drive (super- sedes 609113)
Part Number: 618001
	Group: GROUP— Cylinder
	Description: Assembly - -Complete cylinder block and head "A" unit with valves, valve cover, camshaft and rocker tmechanism, part of coolant inlet section and part of camshaft drive
Part Number: 618046
	Group: GROUP— Cylinder
	Description: Assembly - Complete cylinder block and head "A" unit with valves, valve cover, camshaft and rocker mechanism, part of coolant inlet section and part of camshaft drive
Part Number: 616457
	Group: GROUP— Cylinder
	Description: Block and Head Assembly - Cylinder "A" unit (super- sedes 616805 which superseded 616121)
Part Number: 616873
	Group: GROUP— Cylinder
	Description: Block and Head Assembly - Cylinder "A" unit (super- sedes 615451)
Part Number: 608758
	Group: GROUP— Cylinder
  Figure and Index: (8, 1)
	Description: - Coolant inlet
Part Number: 608782
	Group: GROUP— Cylinder
	Figure and Index: (8, 2)
	Description: Seal - Coolant pipe to elbow and center connection
Part Number: 608760
	Group: GROUP— Cylinder
	Figure and Index: (8, 3)
	Description: Connection Assembly - Coolant inlet cylinder center
Part Number: *608759
	Group: GROUP— Cylinder
	Description: Connettion ;
Part Number: 608547
	Group: GROUP— Cylinder
	Figure and Index: (8, 4)
	Description: Stud - Coolant inlet cylinder center connection ignition wiring harness
Part Number: 608783
	Group: GROUP— Cylinder
	Figure and Index: (8, 5)
	Description: Elbow Assembly (front) - Coolant inlet "A" unit cyl- inder
Part Number: 608761
	Group: GROUP— Cylinder
	Description: Elbow
Part Number: 608547
	Group: GROUP— Cylinder
	Figure and Index: (8, 4)
	Description: Stud - Coolant inlet cylinder front elbow fire ex- tinguishing tube clip
Part Number: 608787
	Group: GROUP— Cylinder
	Figure and Index: (8, 6)
	Description: Elbow Assembly (rear) - Coolant inlet "A" unit cyl- inder
Part Number: *608763
	Group: GROUP— Cylinder
	Description: Elbow
Part Number: 600191
	Group: GROUP— Cylinder
	Figure and Index: (8, 7)
	Description: Coolant inlet cylinder rear elbow ignition wiring harness
Part Number: 604674
	Group: GROUP— Cylinder
	Figure and Index: (8, 8)
	Description: Gasket - Coolant inlet elbow and center to cylinder
Part Number: 601091
	Group: GROUP— Cylinder
	Figure and Index: (8, 9)
	Description: Coolant inlet front elbow retaining nut
Part Number: 601443
	Group: GROUP— Cylinder
	Description: nee inlet elbow and center connection retaining nut
Part Number: 606345
	Group: GROUP— Cylinder
	Description: Lockwasher - Coolant inlet elbow and center con- nection retaining nut
Part Number: 603589
	Group: GROUP— Cylinder
	Figure and Index: (8, 12)
	Description: Niut - Coolant inlet elbow and center connection retaining
Part Number: 608742
	Group: GROUP— Cylinder
	Description: Spacer - Coolant inlet connection elbow to cylinder
Part Number: 4616383
	Group: GROUP— Cylinder
	Figure and Index: (8, 14)
	Description: Block and Liner Assembly - Cylinder NOTE: Temporary release until liner 615836 is available. (supersedes 600344)
Part Number: *600365
	Group: GROUP— Cylinder
	Description: Blok, Tubes and Seals Assembly - Cylinder
Part Number: 600161
	Group: GROUP— Cylinder
	Description: Tube - Cylinder block intermediate stud
Part Number: 600279
	Group: GROUP— Cylinder
	Figure and Index: (8, 16)
	Description: Seal Cylinder block intermediate stud tube
Part Number: 616377
	Group: GROUP— Cylinder
	Description: LinerCylinder block (supersedes 600286 and installed temporarily until liner 615836 is available)
Part Number: 615836
	Group: GROUP— Cylinder
	Description: Liner - -Cylinder block Collar
Part Number: 600287
	Group: GROUP— Cylinder
	Description: Collar Cylinder block to liner lower seal retaining
Part Number: 600289
	Group: GROUP— Cylinder
	Description: liner to block upper
```

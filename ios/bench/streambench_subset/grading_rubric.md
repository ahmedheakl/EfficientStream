# StreamBench subset grading rubric

Grade each `"kind":"question"` row 1 (correct) or 0 (wrong) by comparing `answer` with `reference`. The video is not visible to the grader, so the question is whether the answer conveys the reference's key fact(s).

## Core rules
- **1** if the answer's key content (object, count, colour, action, location, yes/no, name, advice) matches the reference. Extra harmless detail and paraphrase are fine.
- **0** if the answer contradicts the reference, names a different object/count/action, is a refusal or non-answer ("I can't tell", "Sure! I can help..." with no content, "The video does not provide..."), or is too vague to contain the key fact.
- **KG (knowledge) and CI (instructions/advice):** 1 if the answer is substantively consistent with the reference's main point, even when worded differently or longer. 0 if it misses or contradicts that main point.
- **Partial answers:** 1 only if the reference's main fact is present and nothing stated contradicts it.

## Borderline decisions (apply the same way on later passes)
1. **Existence questions** ("Is there X?", "Have you seen X?", "Are there any X in the video?", "Can you find a X in the video?"): a bare correct "Yes"/"No" earns 1 even when the reference also gives a location (ids 11, 94, 124, 140, 157, 170).
2. **Location-seeking questions** ("Where is X?", "Help me find X", "Where did I put X?"): a location is required. A generic location that just restates the setting ("in the kitchen", "in the living room", "in the workshop", "on the floor", "in the background", "on the counter") when the reference names a different or more specific place earns 0 (ids 25, 26, 36, 38, 97, 98, 115, 135, 142, 149, 188). If the generic surface matches the reference's surface, it earns 1: "on the table" for "on the table next to X" (65, 195), "wooden surface" for "wooden table" (29, 31), "dining table" for "white table surrounded by black chairs" (127).
3. **Counter vs table:** "kitchen counter" against a reference "table" is a different surface and earns 0, especially when the reference's distinguishing detail is missing (177, 178).
4. **A correct yes/no with a wrong stated detail earns 0.** Examples: "white apron" when the reference says green (95); "a small plane" when it says helicopter (119, 116); "two stools" when it says wooden chairs (125). Any stated detail must not contradict the reference.
5. **Extra items beyond an exhaustive reference earn 0.** "Green and purple" when the reference says purple (12). "Shrimp, cheese, and pasta" when the reference says the ingredient is shrimp (16). Extra descriptive detail that the reference doesn't cover, and so doesn't contradict, is fine, e.g. "red and blue parrot" for "a parrot" (64).
6. **Two-part questions:** both parts must be right. For "left hand brush, right hand wood" against "brush, metal spatula", the answer earns 0 (156).
7. **Letter answers to unlettered options:** "A." for "a kitchen, a study, or a bedroom" is treated as a non-answer and earns 0 (39).
8. **Hedged lists of generic guesses** ("kitchen, near the stove, or the refrigerator") earn 0 (165). A hedge that clearly implies the correct yes/no ("You should have added some oil") earns 1 (183).
9. **Close action paraphrases** earn 1: "cut" for "shape" with a cutting machine (30), "dig into" for "turn" the soil (63). A different action or object earns 0: "rolling dough" for "spreading sauce" (49), "trimming grass with a trimmer" for "trimming plant branches with shears" (32), "sword" for "thin long stick" (160).
10. **Vague activity or food descriptions** that lack the key fact earn 0: "preparing food" for "cutting green fruit" (184), "fried food" for "pancakes" (141).
11. **KG leniency:** an answer that hits one main point of a multi-point reference earns 1: garlic is anti-inflammatory and antibacterial (45); ventilation against formaldehyde (106); composting kitchen waste (78); "protect food" for plastic wrap (129); "new look, visually appealing" for painting furniture (198); "carpenter" for "a worker using tools" (158). Tautologies or generic filler earn 0: "Bacon is a type of meat" (164), "efficient and safe transportation" for why trains have many cars (174), "food needs to be kept cold" when the reference is about bacteria (181), "ensure safety" as a job guess (69).
12. **Same-video context:** if another reference from the same video settles an ambiguity, it may be used. The mirror is "above the sink" (153 ref), so "above the sink" earns 1 against "above a white table" (154).
13. **Odd or mismatched references:** grade against the reference as written. A non-answer to such a question still earns 0 (107).

## Additional decisions settled during the evict8 pass
14. **Answers identical to an already-graded run** reuse that grade. 47 of the evict8 answers were identical to the compress answers.
15. **Near-synonym surfaces or objects** earn 1: "on the ground" for "on the wooden floor" (7, 163); "desk" for "wooden table" (85); "oven handle" for "stove's railing" (110); "shelf" for "bookcase" (161); "wooden countertop left of the stove" for "wooden board next to the stove" (25); "vanity table" for "white table" (154); "kitchen counter" for "countertop next to the sink" (142); "pressure cooker" for "a pot" (120).
16. **A different object, even a closely related one,** earns 0: "black pot" for "black pot lid" (24); "trowel" for "small metal spatula" (156); "oven" for "microwave" (145); "baking sheet" for "white plate" (19).
17. **Hedged or questioning answers** ("It's on the wall, right? ... I'm trying to pinpoint it", "It's on the counter, right?") earn 0 (3, 130).
18. **Tautologies that restate the question without the key fact** earn 0: "The tool ... is used to paint the ceiling" (72), "can be used for cooking" (122).
19. **Existence questions with an unverifiable extra location** earn 1 when the reference gives no location to contradict: "Yes, there is a clock on the wall" (162). A wrong colour still earns 0: "blue bucket" against "green bucket" (102).
20. **KG answers** must contain the reference's main point. For formaldehyde, "protective clothing and dry paint" without ventilation or plants earns 0 (106). For trains, "transport passengers and cargo efficiently" earns 0 (174), consistent with the compress pass.
21. **Close fruit identification:** "placing figs into a bowl" earns 1 against "cutting open the green fruit and placing it in a bowl" (184).

## Re-check note
The compress grades were re-checked against decisions 14-21 and no call changed. The compress answers that rules 15-19 apply to were already graded the same way: generic "counter"/"kitchen"/"wall" locations, drawer vs cabinet, plane vs helicopter, sword vs stick, and "Yes ... on the table" for an existence question (176). The one hedge that implies yes (183) stays 1 under decision 8.

## Additional decisions settled during the compress4 pass
22. **Answers identical to compress or evict8** reuse that grade. 131 of the 200 compress4 answers were reused this way.
23. **"Center of the room"** earns 1 against "on the wooden floor, in the middle between the TV and the cabinet" because "middle" is the key location (7). "In the corner" still earns 0 against that reference.
24. **Partial action sequences:** "She folded the dough" earns 1 against "folded the dough together and then placed it on the stove" because the main action is present and nothing contradicts it (50). "Rolled the dough out" still earns 0.
25. **KG use questions:** "smelting or metalworking" for a furnace earns 1 against "heating metal to high temperatures" (113). Naming the object without its use earns 0: "it is a blender" for "what can it be used for" (122). A related trade guess counts like "carpenter": "profession related to painting" earns 1 against "a worker using tools" (158).

## Re-check note (decisions 22-25)
The compress and evict8 grades were re-checked against decisions 22-25 and no call changed:
- Decision 23: compress 7 ("In the corner") and compress 190 ("center" against a reference of "corner") stay 0.
- Decision 24: both runs' answer for 50 is "rolled out", a different action, so it stays 0.
- Decision 25: answer 113 is "ironing" in both runs, which is not heating metal, so it stays 0. compress 111 ("Fan") and compress 122 ("a knife") name an object without its use and stay 0. evict8 69 ("meticulous and precise worker") names no trade and stays 0. Both runs answer "carpenter" for 158, which stays 1.

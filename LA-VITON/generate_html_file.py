
import os.path as osp

# load data list
im_names = []
c_names = []

with open(osp.join("D:/Datasets/viton_resize", "test_pairs.txt"), 'r') as f:
    for line in f.readlines():
        im_name, c_name = line.strip().split()  # 000001_0.jpg 001744_1.jpg
        im_names.append(im_name)
        c_names.append(c_name)

with open("cp-vton+_results_comparison.html", "a+") as myfile:
    myfile.write("<html    lang = \"en\">")
    myfile.write("<head>")
    myfile.write("<meta    charset = \"UTF-8\">")
    myfile.write("<title> Compare CP-VTON and CP-VTON+ </title>")
    myfile.write("</head>")
    myfile.write("<body>")
    myfile.write("<table border = \"1\">")
    myfile.write("<thead><tr><td> Image name </td><td> In shop cloth </td><td> warped cloth </td><td> Im head </td> <td> Pose heat map </td><td> body shape </td><td> m_composite </td><td> p_rendered </td><td> Try-on result </td><td> Model </td><td> Segmentation </td><td> Body mask </td></tr></thead>")
    myfile.write("<tbody>")

    for i in range(len(im_names)):

        mask_name = im_names[i].replace(".jpg", ".png")

        myfile.write("<tr>")

        myfile.write("<td>")
        myfile.write(im_names[i] + "<br><b>CP-VTON</b>")
        myfile.write("</td>")

        myfile.write("<td rowspan='2'>")
        myfile.write(
            "<img src = \"D:/Datasets/viton_resize/test/cloth/"+c_names[i]+"\">")
        myfile.write("</td>")

        myfile.write("<td>")
        myfile.write(
            "<img src = \"D:/Datasets/viton_resize/test/warp-cloth1/"+c_names[i]+"\">")
        myfile.write("</td>")

        myfile.write("<td>")
        myfile.write(
            "<img src = \"D:/ThaiTuan/Fashion/cp-vton-master/result/TOM_with_mask_case0_testtest/tom_final.pth/test/im_h/"+im_names[i]+"\">")
        myfile.write("</td>")

        myfile.write("<td rowspan='2'>")
        myfile.write(
            "<img src = \"D:/ThaiTuan/Fashion/cp-vton-master/result/TOM_with_mask_case0_testtest/tom_final.pth/test/im_pose/"+im_names[i]+"\">")
        myfile.write("</td>")

        myfile.write("<td>")
        myfile.write(
            "<img src = \"D:/ThaiTuan/Fashion/cp-vton-master/result/TOM_with_mask_case0_testtest/tom_final.pth/test/shape/" + im_names[i] + "\">")
        myfile.write("</td>")

        myfile.write("<td>")
        myfile.write(
            "<img src = \"D:/ThaiTuan/Fashion/cp-vton-master/result/TOM_with_mask_case0_testtest/tom_final.pth/test/m_composite/"+im_names[i]+"\">")
        myfile.write("</td>")

        myfile.write("<td>")
        myfile.write(
            "<img src = \"D:/ThaiTuan/Fashion/cp-vton-master/result/TOM_with_mask_case0_testtest/tom_final.pth/test/p_rendered/"+im_names[i]+"\">")
        myfile.write("</td>")

        myfile.write("<td>")
        myfile.write(
            "<img src = \"D:/ThaiTuan/Fashion/cp-vton-master/result/TOM_with_mask_case0_testtest/tom_final.pth/test/try-on/" + im_names[i] + "\">")
        myfile.write("</td>")

        myfile.write("<td rowspan='2'>")
        myfile.write(
            "<img src = \"D:/Datasets/viton_resize/test/image/"+im_names[i]+"\">")
        myfile.write("</td>")

        myfile.write("<td>")
        myfile.write(
            "<img src = \"D:/Datasets/viton_resize/test/image-parse/"+mask_name+"\">")
        myfile.write("</td>")

        myfile.write("<td>")
        myfile.write(
            "N/A")
        myfile.write("</td>")

        myfile.write("</tr>")

        # -------------------------------- CP-VTON+ ----------------------------------------------------

        myfile.write("<tr>")

        myfile.write("<td>")
        myfile.write(im_names[i] + "<br><b>CP-VTON+</b>")
        myfile.write("</td>")

        myfile.write("<td>")
        myfile.write(
            "<img src = \"gmm_final.pth/test/warp-cloth/"+c_names[i]+"\">")
        myfile.write("</td>")

        myfile.write("<td>")
        myfile.write(
            "<img src = \"tom_final.pth/test/im_h/" + im_names[i] + "\">")
        myfile.write("</td>")

        myfile.write("<td>")
        myfile.write(
            "<img src = \"tom_final.pth/test/shape/" + im_names[i] + "\">")
        myfile.write("</td>")

        myfile.write("<td>")
        myfile.write(
            "<img src = \"tom_final.pth/test/m_composite/" + im_names[i] + "\">")
        myfile.write("</td>")

        myfile.write("<td>")
        myfile.write(
            "<img src = \"tom_final.pth/test/p_rendered/" + im_names[i] + "\">")
        myfile.write("</td>")

        myfile.write("<td>")
        myfile.write(
            "<img src = \"tom_final.pth/test/try-on/" + im_names[i] + "\">")
        myfile.write("</td>")

        myfile.write("<td>")
        myfile.write(
            "<img src = \"D:/Datasets/viton_resize/test/image-parse-new-vis/" + mask_name + "\">")
        myfile.write("</td>")

        myfile.write("<td>")
        myfile.write(
            "<img src = \"D:/Datasets/viton_resize/test/image-mask/" + mask_name + "\">")
        myfile.write("</td>")

        myfile.write("</tr>")

    myfile.write("</tbody></table></body></html>")

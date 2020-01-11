Summarize my ML projects when 2017 working as internship student

# Project1. DCGAN_and_VGGNet
DCGAN generate apartment images and VGGNet was improvement form CNN for image classification


# DCGAN

I worked DCGAN on jupyter with GPU environment using nvidia Docker. DCGAN generate apartment images.


Its aim is to generate dataset images for image classification project.

To know its mechanism and process, please see below site(my blog)

http://trafalbad.hatenadiary.jp/entry/2017/10/28/223421




# VGGNet(CNN improvement)


I made VGGNet by improving simple  CNN for image classification project.

VGGNet can classify clean images (kitchen, Landscape, Living room) and dirty image (toilet) that are web site images.

This is because visitor to the web site can see clean image at frist page and will not go back from the web site because of good impression.

The project detail is written in below site(my blog)


http://trafalbad.hatenadiary.jp/entry/2017/09/30/142505





# Project2. Naivebays_and_Doc2vec
Naive bays model to clean apartment name and Doc2vec model to display similar articles for SEO


# Naivebays model
<b>【description】</b>


⑴. I scraped apartment names with noise from Internet (just like 「【値下げ！】弊社限定未公開物件！豊田駅徒歩８分 ライオンズガーデン明大前パラダイム), making dataset with mecab libarary.(dataset.csv)

⑵. I converted the txets to Bag-of-words and trained by Naive bays (making_app_process.py).

⑶. I made application with libarary named 'bottle' which send the apartment mame with noise from client and get response clean apartment name without noise from server just like below(bottle_application.py)

```
# client's request (apartment name with noise)
【値下げ！】弊社限定未公開物件！豊田駅徒歩８分 ライオンズガーデン明大前パラダイム

# response from server (clean apartment name without noise)
ライオンズガーデン明大前パラダイム
```
⑷.「Naivebays_poject_presentation.docx」indicates details of this project






# Doc2vec model
<b>【description】</b>


By using 1659 articles in a blog, Doc2vec model predict most similar 10 articles with a article from the 1659 articles for SEO efforts.

<b>【main point】</b>


⑴.「Doc2vec_train.py」indicates some technics to improve accuracy

→ only use particle, adjective and verb in sentence for train

→delete unnecessary words in article sentence.

→used library is mecab.

⑵.Finally, similarly rates of every articles I predicted were saved to csv file (making_csv.py). The csv file is「similar_rate.csv」



⑶.「presentation_docs-2017.pdf」and 「Doc2vec_poject_doc.pdf」indicates details of this project.

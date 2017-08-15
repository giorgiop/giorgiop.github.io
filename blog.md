---
layout: page
title: Blog
---

<br>

<div class="posts">
  {% for post in site.posts %}
    <article class="post">

      <h3><a href="{{ post.url }}">{{ post.title }}</a></h3>
      <span class="post-date">{{ post.date | date_to_string }}</span>

      <div class="entry">
        {{ post.summary }}
      </div>

    </article>
  {% endfor %}
</div>

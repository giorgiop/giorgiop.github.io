---
layout: page
title: Blog
---

<br>

<div class="posts">
<ul>
  {% for post in site.posts %}

  <li>

      <h4>
      {{ post.date | date_to_string }} |
      <a href="{{ post.url }}">{{ post.title }}</a>
      </h4>



        {{ post.summary }}


    </li>
  {% endfor %}
  </ul>
</div>
